import torch
from torch import nn
from torch_scatter import scatter

from ocpmodels.models.gemnet.layers.base_layers import Dense, ResidualLayer
from ocpmodels.modules.scaling.scale_factor import ScaleFactor
from ocpmodels.equiformer_v2.so3 import(
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
    CoefficientMappingModule
)
from ocpmodels.equiformer_v2.activation import GateActivation


class EwaldBlock(torch.nn.Module):
    """
    Long-range block from the Ewald message passing method

    Parameters
    ----------
        shared_downprojection: Dense,
            Downprojection block in Ewald block update function,
            shared between subsequent Ewald Blocks.
        emb_size_atom: int
            Embedding size of the atoms.
        downprojection_size: int
            Dimension of the downprojection bottleneck
        num_hidden: int
            Number of residual blocks in Ewald block update function.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
        use_pbc: bool
            Set to True if periodic boundary conditions are applied.
        delta_k: float
            Structure factor voxel resolution
            (only relevant if use_pbc == False).
        k_rbf_values: torch.Tensor
            Pre-evaluated values of Fourier space RBF
            (only relevant if use_pbc == False).
        return_k_params: bool = True,
            Whether to return k,x dot product and damping function values.
    """

    def __init__(
        self,
        shared_downprojection: Dense,
        emb_size_atom: int,
        downprojection_size: int,
        num_hidden: int,
        lmax_list: list,
        mmax_list: list,
        channel: int,
        activation=None,
        name=None,  # identifier in case a ScalingFactor is applied to Ewald output
        use_pbc: bool = True,
        delta_k: float = None,
        k_rbf_values: torch.Tensor = None,
        return_k_params: bool = True,
    ):
        super().__init__()
        self.use_pbc = use_pbc
        self.return_k_params = return_k_params

        self.delta_k = delta_k
        self.k_rbf_values = k_rbf_values

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.channel = channel
        self.l_max =  lmax_list[0]
        self.down = shared_downprojection
        self.up = Dense(
            downprojection_size, emb_size_atom, activation=None, bias=False
        )
        # self.pre_residual = ResidualLayer(
        #     channel, nLayers=2, activation=activation
        # )
        self.residual = SO3_LinearV2(self.channel, self.channel, lmax=self.lmax_list[0])
        # self.irreps_residual = nn.ModuleList([
        #     ResidualLayer(self.channel, nLayers=2, activation=activation)
        #     for _ in range(0, self.l_max + 1)])
        # # [, 2l+1, channel]


        self.ewald_layers = nn.ModuleList([
            self.get_mlp(emb_size_atom, emb_size_atom, num_hidden, activation)
            for _ in range(self.l_max + 1)
        ])


        self.gate0_linear_1 = nn.Linear(self.channel, max(self.lmax_list) * self.channel)
        self.gate0_linear_2 = nn.Linear(self.channel, max(self.lmax_list) * self.channel)

        self.gate_act_1 = GateActivation(
            lmax=max(self.lmax_list), 
            mmax=max(self.mmax_list), 
            num_channels=self.channel
        )

        self.gate_act_2 = GateActivation(
            lmax=max(self.lmax_list), 
            mmax=max(self.mmax_list), 
            num_channels=self.channel
        )
        
        # self.ewald_layers = self.get_mlp(
        #     emb_size_atom, emb_size_atom, num_hidden, activation
        # )
        if name is not None:
            self.ewald_scale_sum = ScaleFactor(name + "_sum")
        else:
            self.ewald_scale_sum = None

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(num_hidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(
        self,
        h: torch.Tensor,    # embedding
        x: torch.Tensor,    # positions
        k: torch.Tensor,    
        num_batch: int,
        batch_seg: torch.Tensor,
        # Dot products k^Tx and damping values: need to be computed only once per structure
        # Ewald block in first interaction block gets None as input, therefore computes these
        # values and then passes them on to Ewald blocks in later interaction blocks
        dot: torch.Tensor = None,
        sinc_damping: torch.Tensor = None,
    ):
        '''    
        h_ewald, dot, sinc_damping = self.ewald_blocks[i](
            h,
            pos,
            k_grid,
            batch_size,
            batch,
            dot,
            sinc_damping,
        )'''

        res_embedding = self.residual(h)
        x_0_gating = res_embedding.embedding[:, 0, :]
        x_0_gating = self.gate0_linear_1(x_0_gating)  # [N, lmax * C]
        res_embedding.embedding = self.gate_act_1(x_0_gating, res_embedding.embedding)
        h_update_list = []
        res_embed = res_embedding.embedding  # [N, total_M, C]
        N, total_M, C = res_embed.shape

        # dot product k^Tx
        if dot is None:
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])  # [N, K, C]
            dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)  # [N, K]

        if sinc_damping is None:
            if not self.use_pbc:
                sinc_damping = (
                    torch.sinc(0.5 * self.delta_k * x[:, 0].unsqueeze(-1)) *
                    torch.sinc(0.5 * self.delta_k * x[:, 1].unsqueeze(-1)) *
                    torch.sinc(0.5 * self.delta_k * x[:, 2].unsqueeze(-1))
                )  # [N, K]
                sinc_damping = sinc_damping.expand(-1, k.shape[-2])  # broadcast
            else:
                sinc_damping = 1  # scalar or tensor of shape [N, K]

        cos_dot = torch.cos(dot).unsqueeze(-1).unsqueeze(-1)
        sin_dot = torch.sin(dot).unsqueeze(-1).unsqueeze(-1)
        damping = sinc_damping if isinstance(sinc_damping, torch.Tensor) else 1.
        damping = damping.unsqueeze(-1).unsqueeze(-1)

        if self.use_pbc:
            base_filter = torch.matmul(self.up.linear.weight, self.down.linear.weight).T
            k_filter_pbc = base_filter.unsqueeze(0).expand(num_batch, -1, -1)
        else:
            self.k_rbf_values = self.k_rbf_values.to(x.device)
            proj = self.down(self.k_rbf_values)
            k_out = self.up(proj)  # [K, C]

        offset = 0
        for l in range(self.l_max + 1):
            M = 2 * l + 1
            hres_l = res_embed[:, offset:offset + M, :]
            offset += M

            # --- Build k_filter ---
            if self.use_pbc:
                k_filter = k_filter_pbc
            else:
                k_filter = k_out.unsqueeze(1).expand(-1, M, -1)

            # --- Structure factors ---
            h_exp = hres_l.unsqueeze(1)  # [N, 1, M, C]
            sf_real = torch.zeros(num_batch, dot.shape[1], M, C, device=hres_l.device).index_add_(
                0, batch_seg, h_exp * cos_dot * damping
            )
            sf_imag = torch.zeros_like(sf_real).index_add_(
                0, batch_seg, h_exp * sin_dot * damping
            )

            # --- Fourier filter ---
            if self.use_pbc:
                sf_real = torch.matmul(sf_real, k_filter)
                sf_imag = torch.matmul(sf_imag, k_filter)
            else:
                sf_real = sf_real * k_filter.unsqueeze(0)
                sf_imag = sf_imag * k_filter.unsqueeze(0)

            # --- Back to atom space ---
            real_part = torch.index_select(sf_real, 0, batch_seg)
            imag_part = torch.index_select(sf_imag, 0, batch_seg)
            h_update_l = 0.01 * torch.sum((real_part * cos_dot + imag_part * sin_dot) * damping, dim=1)

            for layer in self.ewald_layers[l]:
                h_update_l = layer(h_update_l)

            h_update_list.append(h_update_l)

        
        all_embeddings = torch.cat(h_update_list, dim=1) 
        h_so3 = SO3_Embedding(
            0,
            lmax_list=self.lmax_list,
            num_channels=self.channel,
            device = res_embed.device,
            dtype = res_embed.dtype,
        )
        h_so3.set_embedding(all_embeddings)
        h_so3.set_lmax_mmax(self.l_max, self.l_max)

        # gate activation function
        x_0_gating_2 = h_so3.embedding[:, 0, :]
        x_0_gating_2 = self.gate0_linear_2(x_0_gating_2)  # [N, lmax * C]
        h_output = self.gate_act_2(x_0_gating_2, h_so3.embedding)

        # h_output = SO3_Embedding(
        #     0,
        #     lmax_list=self.lmax_list,
        #     num_channels=self.channel,
        #     device=res_embed.device,
        #     dtype=res_embed.dtype,
        # )
        # h_output.set_embedding(h_output.embedding)
        # h_output.set_lmax_mmax(self.l_max, self.l_max)
        # if self.ewald_scale_sum is not None:
        #     h_update = self.ewald_scale_sum(h_update, ref=h)

        # # Apply update function
        # for layer in self.ewald_layers:
        #     h_update = layer(h_update)

        if self.return_k_params:
            return h_output, dot, sinc_damping
        else:
            return h_output


# Atom-to-atom continuous-filter convolution
class HadamardBlock(torch.nn.Module):
    """
    Aggregate atom-to-atom messages by Hadamard (i.e., component-wise)
    product of embeddings and radial basis functions

    Parameters
    ----------
        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_atom: int
            Embedding size of the edges.
        nHidden: int
            Number of residual blocks.
        activation: callable/str
            Name of the activation function to use in the dense layers.
        scale_file: str
            Path to the json file containing the scaling factors.
        name: str
            String identifier for use in scaling file.
    """

    def __init__(
        self,
        emb_size_atom: int,
        emb_size_bf: int,
        nHidden: int,
        activation=None,
        scale_file=None,
        name: str = "hadamard_atom_update",
    ):
        super().__init__()
        self.name = name

        self.dense_bf = Dense(
            emb_size_bf, emb_size_atom, activation=None, bias=False
        )
        self.scale_sum = ScalingFactor(
            scale_file=scale_file, name=name + "_sum"
        )
        self.pre_residual = ResidualLayer(
            emb_size_atom, nLayers=2, activation=activation
        )
        self.layers = self.get_mlp(
            emb_size_atom, emb_size_atom, nHidden, activation
        )

    def get_mlp(self, units_in, units, nHidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for i in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(self, h, bf, idx_s, idx_t):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size_atom)
                Atom embedding.
        """
        nAtoms = h.shape[0]
        h_res = self.pre_residual(h)

        mlp_bf = self.dense_bf(bf)

        x = torch.index_select(h_res, 0, idx_s) * mlp_bf

        x2 = scatter(x, idx_t, dim=0, dim_size=nAtoms, reduce="sum")
        # (nAtoms, emb_size_edge)
        x = self.scale_sum(h, x2)

        for layer in self.layers:
            x = layer(x)  # (nAtoms, emb_size_atom)

        return x



    #    embedding = h.embedding
    #     # dot product k^Tx
    #     if dot is None:
    #         b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])  # [N, K, C]
    #         dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)  # [N, K]

    #     if sinc_damping is None:
    #         if not self.use_pbc:
    #             sinc_damping = (
    #                 torch.sinc(0.5 * self.delta_k * x[:, 0].unsqueeze(-1)) *
    #                 torch.sinc(0.5 * self.delta_k * x[:, 1].unsqueeze(-1)) *
    #                 torch.sinc(0.5 * self.delta_k * x[:, 2].unsqueeze(-1))
    #             )  # [N, K]
    #             sinc_damping = sinc_damping.expand(-1, k.shape[-2])  # broadcast
    #         else:
    #             sinc_damping = 1  # scalar or tensor of shape [N, K]
        
    #     res_embedding = self.residual(h)
    #     print("res_embedding.shape", res_embedding.embedding.shape)

    #     for l in range(self.l_max + 1):
    #         # Residual connection
    #         h_l = h[l]
    #         hres_l = self.irreps_residual[l](h_l)

    #         # Fourier space filter
    #         if self.use_pbc:
    #             # k_filter: [B, C, C] for all l
    #             base_filter = torch.matmul(self.up.linear.weight, self.down.linear.weight).T
    #             k_filter = base_filter.unsqueeze(0).expand(num_batch, -1, -1)  # [B, C, C]
    #         else:
    #             self.k_rbf_values = self.k_rbf_values.to(x.device)
    #             proj = self.down(self.k_rbf_values)  # [K, D_down]
    #             k_out = self.up(proj)                # [K, C]
    #             k_filter = k_out.unsqueeze(1).expand(-1, 2 * l + 1, -1)  # [K, 2l+1, C]

    #         # tructure factors
    #         N, M, C = hres_l.shape
    #         cos_dot = torch.cos(dot).unsqueeze(-1).unsqueeze(-1)  # [N, K, 1, 1]
    #         sin_dot = torch.sin(dot).unsqueeze(-1).unsqueeze(-1)
    #         damping = sinc_damping if isinstance(sinc_damping, torch.Tensor) else 1.
    #         damping = damping.unsqueeze(-1).unsqueeze(-1)  # [N, K, 1, 1]
    #         h_exp = hres_l.unsqueeze(1)  # [N, 1, M, C]

    #         sf_real = torch.zeros(num_batch, dot.shape[1], M, C, device=hres_l.device).index_add_(
    #             0, batch_seg, h_exp * cos_dot * damping
    #         )
    #         sf_imag = torch.zeros_like(sf_real).index_add_(
    #             0, batch_seg, h_exp * sin_dot * damping
    #         )

    #         # Fourier filter 
    #         if self.use_pbc:
    #             sf_real = torch.matmul(sf_real, k_filter)  # [B, K, 2l+1, C]
    #             sf_imag = torch.matmul(sf_imag, k_filter)
    #         else:
    #             sf_real = sf_real * k_filter.unsqueeze(0)  # [B, K, 2l+1, C]
    #             sf_imag = sf_imag * k_filter.unsqueeze(0)

    #         # Back to atom space
    #         real_part = torch.index_select(sf_real, 0, batch_seg)  # [N, K, 2l+1, C]
    #         imag_part = torch.index_select(sf_imag, 0, batch_seg)

    #         cos_dot = torch.cos(dot).unsqueeze(-1).unsqueeze(-1)
    #         sin_dot = torch.sin(dot).unsqueeze(-1).unsqueeze(-1)
    #         damping = sinc_damping.unsqueeze(-1).unsqueeze(-1) if isinstance(sinc_damping, torch.Tensor) else 1.

    #         h_update_l = 0.01 * torch.sum(
    #             (real_part * cos_dot + imag_part * sin_dot) * damping, dim=1
    #         )  # [N, 2l+1, C]

    #         for layer in self.ewald_layers[l]:
    #             h_update_l = layer(h_update_l)
    #         h_update_list.append(h_update_l)
