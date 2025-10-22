import torch
from typing import Union, Optional
import matplotlib.pyplot as plt


class Coil:
    """Class for cyclindrical coil with multiple channels placed on rings."""
    def __init__(
        self,
        coil_radius: float, 
        num_channels: int = 32,
        num_rings: int = 4,
        ring_spacing: float = 0.05,
        loop_radius: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize a Coil instance.

        Parameters
        ----------
        coil_radius : float
            The radius of the coil in meter.
        num_channels : int, optional
            The number of coil channels, by default 32.
        num_rings : int, optional
            The number of rings in the coil, by default 4.
        ring_spacing : float, optional
            The spacing between rings in meter, by default 0.05.
        """
        if device is None:
            device = torch.device('cpu')
            
        if num_channels % num_rings != 0:
            raise ValueError("num_channels must be divisible by num_rings.")
        
        self.coil_radius = coil_radius
        self.num_channels = num_channels
        self.num_rings = num_rings
        self.ring_spacing = ring_spacing
        self.loop_radius = loop_radius
        self.device = device
        self.num_loop_elements = 20

        # Create channel loops
        ring_positions = (torch.arange(num_rings) - (num_rings-1)/2) * ring_spacing
        num_channels_per_ring = num_channels // num_rings
        phi_offset = torch.pi / num_channels_per_ring  # staggered arrangement
        dphi_ring = 2 * torch.pi / num_channels_per_ring  # angular spacing per ring
        
        self.coil_loops : list[CoilLoop] = [] 
        for enum, rp in enumerate(ring_positions):
            for ec in range(num_channels_per_ring):
                phi = torch.tensor(ec * dphi_ring + (enum % 2) * phi_offset)
                coil_normal = torch.tensor(
                    [torch.cos(phi), torch.sin(phi), 0.0], device=self.device
                )
                coil_center = torch.tensor(
                    [self.coil_radius * torch.cos(phi), 
                     self.coil_radius * torch.sin(phi), 
                     rp],
                    device=self.device
                )
                coil_loop = CoilLoop(
                    center_loc=coil_center,
                    normal=coil_normal,
                    radius=self.loop_radius,
                    num_elements=self.num_loop_elements
                )
                self.coil_loops.append(coil_loop)


    @property
    def coil_channel_centers(self):
        """Get the coil channel center positions."""
        return torch.stack([coil.center_loc for coil in self.coil_loops], dim=0)
    
    @property
    def coil_channels_elements(self):
        return torch.stack([loop.elements_locs for loop in self.coil_loops], dim=0)
    

    def get_sensitivity_maps(self, matrix_size, fov, chunk_size=10000):
        rl_batch = torch.stack(
            [c.elements_locs.T.contiguous() for c in self.coil_loops],
            dim=0
        ).to(self.device)
        
        dl_batch = torch.stack(
            [c.elements_dls.T.contiguous() for c in self.coil_loops],
            dim=0
        ).to(self.device)
        
        dphi = self.coil_loops[0].dphi 
    
        sample_locs, reshape_fn = _make_grid(matrix_size, fov, device=self.device)

        B = _biot_savart_batched(
            sample_locs,
            rl_batch,
            dl_batch,
            dphi,
            chunk_size=chunk_size
        )

        sens_flat = B[:, 0, :] - 1j * B[:, 1, :]

        sens_maps = torch.stack(
            [reshape_fn(sens_flat[c]) for c in range(B.shape[0])],
            dim=0
        )
        
        return sens_maps
    
    def plot_coil_geometry(self):
        """Plot the coil geometry using matplotlib."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for coil in self.coil_loops:
            locs = coil.elements_locs.cpu().numpy()
            ax.plot(locs[0], locs[1], locs[2])

        ax.set_xlim(-self.coil_radius - self.loop_radius, self.coil_radius + self.loop_radius)
        ax.set_ylim(-self.coil_radius - self.loop_radius, self.coil_radius + self.loop_radius)
        ax.set_zlim(-self.coil_radius - self.loop_radius, self.coil_radius + self.loop_radius)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Coil Geometry')
        plt.show()

class CoilLoop:
    """Class representing a single coil loop."""
    def __init__(
        self,
        center_loc: Union[torch.Tensor, tuple[float, ...], list[float]],
        normal: Union[torch.Tensor, tuple[float, ...], list[float]],
        radius: Union[float, torch.Tensor],
        num_elements: int = 40,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize a CoilLoop instance.

        Parameters
        ----------
        center_loc : Union[torch.Tensor, tuple[float, ...], list[float]]
            The (x, y, z) coordinates of the coil loop center in meter.
        normal : Union[torch.Tensor, tuple[float, ...], list[float]]
            The normal vector of the coil loop plane.
        radius : Union[float, torch.Tensor]
            The radius of the coil loop in meter.
        num_elements : int, optional
            The number of discrete elements to represent the coil loop, by default 40.
        """
        if device is None:
            device = torch.device('cpu')
        self.device = device 

        if isinstance(center_loc, (tuple, list)):
            center_loc = torch.tensor(center_loc, dtype=torch.float32, device=self.device)
        if isinstance(normal, (tuple, list)):
            normal = torch.tensor(normal, dtype=torch.float32, device=self.device)
        if isinstance(radius, float):
            radius = torch.tensor(radius, dtype=torch.float32, device=self.device)

        self.center_loc = center_loc
        self.normal = normal / torch.linalg.norm(normal)  # Normalize the normal vector
        self.radius = radius
        self.num_elements = num_elements
        self.phi= torch.linspace(0, 2 * torch.pi, num_elements)
        self.dphi = self.phi[1] - self.phi[0]

        self.elements_locs, self.elements_dls = self.build_loop_elements()


    def build_loop_elements(self):
        """Build the discrete elements representing the coil loop."""
        normal0 = torch.tensor([1.0, 0.0, 0.0])
        # More efficient vectorized version
        dl0_x = torch.zeros_like(self.phi)
        dl0_y = -torch.sin(self.phi) * self.radius  
        dl0_z = torch.cos(self.phi) * self.radius
        dl0 = torch.stack([dl0_x, dl0_y, dl0_z], dim=0)
        
        rphi0_x = torch.zeros_like(self.phi)
        rphi0_y = torch.cos(self.phi) * self.radius
        rphi0_z = torch.sin(self.phi) * self.radius
        rphi0 = torch.stack([rphi0_x, rphi0_y, rphi0_z], dim=0)

        rotmat = _rotation_matrix(normal0, self.normal)
        #rotmat = torch.eye(3)

        loop_element_locs = (rotmat @ rphi0) + self.center_loc[:, None]
        loop_element_dls = (rotmat @ dl0)

        return loop_element_locs, loop_element_dls

    def get_sensitivity(self, sample_locs: torch.Tensor) -> torch.Tensor:
        """Calculate the coil sensitivity at given sample locations.

        Parameters
        ----------
        sample_locs : torch.Tensor
            The (3, N) tensor of sample locations where the sensitivity is evaluated.

        Returns
        -------
        torch.Tensor
            The (N, ) tensor of coil sensitivities at the sample locations.
        """
        mag = _biot_savart(sample_locs, self.elements_locs, self.elements_dls, self.dphi)

        sens = mag[0] - 1j * mag[1]
        
        return sens
    
    def get_sensitivity_map(
        self,
        matrix_size: Union[torch.Tensor, tuple[int, ...], list[int]],
        fov: Union[torch.Tensor, tuple[float, ...], list[float]]
    ) -> torch.Tensor:
        """Calculate the coil sensitivity map over a 3D grid.

        Parameters
        ----------
        matrix_size : Union[torch.Tensor, tuple[int, ...], list[int]]
            The size of the output sensitivity map (nx, ny, nz).
        fov : Union[torch.Tensor, tuple[float, ...], list[float]]
            The field of view in meters (fx, fy, fz).

        Returns
        -------
        torch.Tensor
            The (nx, ny, nz) tensor of coil sensitivities.
        """
        if isinstance(matrix_size, (tuple, list)):
            matrix_size = torch.tensor(matrix_size, dtype=torch.int32)
        if isinstance(fov, (tuple, list)):
            fov = torch.tensor(fov, dtype=torch.float32)
        
        # Calculate at the center of each voxel
        pd = fov / matrix_size
        x = torch.arange(matrix_size[0].item(), dtype=torch.float32, device=self.device) * pd[0] - fov[0]/2 + pd[0]/2
        y = torch.arange(matrix_size[1].item(), dtype=torch.float32, device=self.device) * pd[1] - fov[1]/2 + pd[1]/2
        z = torch.arange(matrix_size[2].item(), dtype=torch.float32, device=self.device) * pd[2] - fov[2]/2 + pd[2]/2
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        sample_locs = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=0)
        
        sens_map = self.get_sensitivity(sample_locs)
        sens_map = sens_map.reshape(matrix_size.tolist())

        return sens_map

def _rotation_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the rotation matrix that rotates vector a to vector b."""
    v = torch.linalg.cross(a, b)
    c = torch.dot(a, b)
    s = torch.norm(v)
    
    if s == 0:
        return torch.eye(3)  # No rotation needed
    
    v_x = torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], device=a.device)
    
    R = torch.eye(3, device=a.device) + v_x + v_x @ v_x * ((1 - c) / (s ** 2))
    
    return R

def _biot_savart(r, rl, dl, dphi):
    """Calculates the biot savart law for wire elements dl at locations rl for point r.
    """
    num_dimensions, num_samples = r.shape
    n_phi = rl.shape[1]
    
    B = torch.zeros((num_dimensions, num_samples), dtype=torch.complex64, device=r.device)
    for i in range(n_phi):
        r_diff = r - rl[:, i][:, None]
        cross = torch.linalg.cross(dl[:, i][None, :], r_diff.T).T
        norm = torch.linalg.norm(r_diff, dim=0)**3
        B += cross/norm[None, :] * dphi
    
    return B


def _make_grid(
    matrix_size: Union[torch.Tensor, tuple[int, ...], list[int]], 
    fov: Union[torch.Tensor, tuple[float, ...], list[float]], 
    device: Optional[torch.device] = None, 
    dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, callable]:
    """Create a 3D grid of sample locations and a reshape function.

    Parameters
    ----------
    matrix_size : Union[torch.Tensor, tuple[int, ...], list[int]]
        The size of the grid (nx, ny, nz).
    fov : Union[torch.Tensor, tuple[float, ...], list[float]]
        The field of view in meters (fx, fy, fz).
    device : Optional[torch.device], optional
        The device for tensor operations, by default None.
    dtype : torch.dtype, optional
        The data type for tensors, by default torch.float32.

    Returns
    -------
    tuple[torch.Tensor, callable]
        A tuple containing:
        - sample_locs: Tensor of shape (3, N) with flattened grid coordinates
        - reshape_fn: Function to reshape flattened arrays back to grid shape
    """
    if isinstance(matrix_size, (tuple, list)):
        matrix_size = torch.tensor(matrix_size, dtype=torch.int32)
    if isinstance(fov, (tuple, list)):
        fov = torch.tensor(fov, dtype=dtype)
    if device is None:
        device = fov.device

    # voxel centers
    pd = fov / matrix_size
    axes = [
        torch.linspace(-fov[i]/2 + pd[i]/2, fov[i]/2 - pd[i]/2, int(matrix_size[i]),
                       device=device, dtype=dtype)
        for i in range(3)
    ]
    X, Y, Z = torch.meshgrid(*axes, indexing='ij')
    sample_locs = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=0)

    def reshape_fn(vec):
        return vec.reshape(tuple(int(n) for n in matrix_size.tolist()))

    return sample_locs, reshape_fn


def _biot_savart_batched(
    r: torch.Tensor,
    rl_batch: torch.Tensor,
    dl_batch: torch.Tensor,
    dphi: torch.Tensor,
    eps: float = 1e-12,
    chunk_size: int = 100000
) -> torch.Tensor:
    """Calculates a batched biot-savart law for multiple coil loops.

    Parameters
    ----------
    r : torch.Tensor
        Sample locations tensor of shape (3, N) where N is the number of sample points.
    rl_batch : torch.Tensor
        Batch of coil loop element positions of shape (C, Ne, 3) where C is the number 
        of coils and Ne is the number of elements per loop.
    dl_batch : torch.Tensor
        Batch of coil loop element differential lengths of shape (C, Ne, 3).
    dphi : torch.Tensor
        Angular spacing between elements in radians.
    eps : float, optional
        Small value to prevent division by zero, by default 1e-12.
    chunk_size : int, optional
        Number of sample points to process at once for memory efficiency, by default 100000.

    Returns
    -------
    torch.Tensor
        Magnetic field tensor of shape (C, 3, N) representing the 3D magnetic field 
        components for each coil at each sample location.
    """
    device = r.device
    dtype  = r.dtype
    C, Ne, _ = rl_batch.shape
    N = r.shape[1]

    out = torch.zeros(C, 3, N, device=device, dtype=dtype)

    rl_e = rl_batch.unsqueeze(2)
    dl_e = dl_batch.unsqueeze(2)

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        
        r_chunk = r[:, s:e].T.unsqueeze(0).unsqueeze(0)

        
        rdiff = r_chunk - rl_e

        # |r|^2 and 1/|r|^3
        norm2 = (rdiff * rdiff).sum(dim=3).clamp_min(eps)
        invn3 = 1.0 / (norm2 * torch.sqrt(norm2))

        # cross(dl, rdiff) component-wise
        cx = dl_e[..., 1]*rdiff[..., 2] - dl_e[..., 2]*rdiff[..., 1]
        cy = dl_e[..., 2]*rdiff[..., 0] - dl_e[..., 0]*rdiff[..., 2]
        cz = dl_e[..., 0]*rdiff[..., 1] - dl_e[..., 1]*rdiff[..., 0]

        out[:, 0, s:e] = (cx * invn3).sum(dim=1) * dphi
        out[:, 1, s:e] = (cy * invn3).sum(dim=1) * dphi
        out[:, 2, s:e] = (cz * invn3).sum(dim=1) * dphi

    return out