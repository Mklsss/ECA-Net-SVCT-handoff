import torch
import numpy as np
import config
import time
import os

from scipy.interpolate import interp1d

ASTRA_INSTALLED = False
try:
    import astra
    ASTRA_INSTALLED = True
    print("ASTRA Toolbox found. Using ASTRA for reconstruction.")
except ImportError:
    print("Error: ASTRA Toolbox not found. Cannot perform reconstruction.")
    print("Please install it: conda install -c astra-toolbox astra-toolbox")
    ASTRA_INSTALLED = False



def fbp_reconstruct(sinogram_batch, angles_rad, output_size):
  
    if not ASTRA_INSTALLED:
        raise RuntimeError("ASTRA Toolbox is required for reconstruction but not found.")

    if sinogram_batch.ndim != 4 or sinogram_batch.shape[1] != 1:
        raise ValueError(f"fbp_reconstruct expects input shape (N, 1, H, W), got {sinogram_batch.shape}")

    batch_size = sinogram_batch.shape[0]
    num_angles_in_sino = sinogram_batch.shape[2] 
    num_detectors = sinogram_batch.shape[3]
    device = sinogram_batch.device

    # --- 检查配置和输入是否匹配 ---
    if config.GEOMETRY != 'fanflat':
        print(f"Warning: Config geometry is '{config.GEOMETRY}', but this function uses 'fanflat'.")
    if num_detectors != config.DETECTORS_COUNT: # DETECTORS_COUNT from config is used for proj_geom
        print(f"Warning: Input has {num_detectors} detectors, config has {config.DETECTORS_COUNT}. Using config value for geometry setup.")
    

    if len(angles_rad) != num_angles_in_sino:
        print(f"Warning: Number of angles in sinogram ({num_angles_in_sino}) does not match "
              f"length of provided angles_rad array ({len(angles_rad)}). This might lead to errors.")
    
    theta = angles_rad 

    output_size_int = int(output_size)

    # --- 定义几何 
    vol_geom = astra.create_vol_geom(output_size_int, output_size_int)
    proj_geom = astra.create_proj_geom(
        'fanflat',
        config.DETECTOR_WIDTH,
        config.DETECTORS_COUNT, 
        theta,                  
        config.DSO,
        config.DSD - config.DSO
    )
   
    if num_detectors != config.DETECTORS_COUNT:
        print(f"Critical Warning: Sinogram detector count ({num_detectors}) "
              f"differs from ASTRA geometry detector count ({config.DETECTORS_COUNT}). "
              f"This will likely cause ASTRA errors or incorrect reconstruction.")


    # --- 重建循环 ---
    reconstructed_images = []
    gpu_failed_once = False
    cpu_used_once = False

    for b in range(batch_size):
        sino_np = sinogram_batch[b, 0].detach().cpu().numpy()

        proj_id = None
        rec_id = None
        alg_id = None
        cfg = None
        reconstruction = None 
        recon_successful = False

        try:
            #  GPU 重建 ---
            proj_id = astra.data2d.create('-sino', proj_geom, sino_np)
            rec_id = astra.data2d.create('-vol', vol_geom)

            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = proj_id
      

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            reconstruction = astra.data2d.get(rec_id)
            recon_successful = True

        except Exception as e_gpu:
            if not gpu_failed_once:
                print(f"\nWarning: ASTRA GPU FBP ('FBP_CUDA') failed ({e_gpu}). Falling back to ASTRA CPU FBP ('FBP'). This will be slower.")
                gpu_failed_once = True

            if alg_id is not None: astra.algorithm.delete(alg_id); alg_id = None

            try:
                # --- 2. 尝试 CPU 重建 ---
                if proj_id is None: # If GPU step failed before creating it
                    proj_id = astra.data2d.create('-sino', proj_geom, sino_np)
                if rec_id is None:
                    rec_id = astra.data2d.create('-vol', vol_geom)

                cfg = astra.astra_dict('FBP')
                cfg['ReconstructionDataId'] = rec_id
                cfg['ProjectionDataId'] = proj_id
                # cfg['FilterType'] = 'Ram-Lak'

                alg_id = astra.algorithm.create(cfg)
                astra.algorithm.run(alg_id)
                reconstruction = astra.data2d.get(rec_id)
                recon_successful = True
                if not cpu_used_once:
                    print("Using ASTRA CPU FBP.", end='\r')
                    cpu_used_once = True

            except Exception as e_cpu:
                print(f"\n--- ASTRA CPU FBP also failed (batch item {b}) ---")
                print(f"Error: {e_cpu}")
                recon_successful = False
        finally:
            if alg_id is not None: astra.algorithm.delete(alg_id)
            if rec_id is not None : astra.data2d.delete(rec_id); rec_id = None 


        if recon_successful and reconstruction is not None:
            min_val = reconstruction.min()
            max_val = reconstruction.max()
            if max_val > min_val:
                reconstruction = (reconstruction - min_val) / (max_val - min_val)
            else:
                reconstruction = np.zeros_like(reconstruction)

            reconstruction = np.flip(reconstruction, axis=0)
            reconstruction = np.rot90(reconstruction, k=-1)
            reconstructed_images.append(torch.from_numpy(reconstruction.copy()).unsqueeze(0).unsqueeze(0))
        else:
            print(f"Reconstruction failed for item {b}, appending zeros.")
            zero_recon = torch.zeros((1, 1, output_size_int, output_size_int), dtype=torch.float32)
            reconstructed_images.append(zero_recon)

    if cpu_used_once: print("") 

    reconstructed_tensor = torch.cat(reconstructed_images, dim=0).to(device)
    return reconstructed_tensor

# --- 保留测试模块，更新以反映 fanflat ---
if __name__ == '__main__':
    print("--- Running reconstruction.py ASTRA fanflat test ---")
    if not ASTRA_INSTALLED:
        print("ASTRA not available, cannot run test.")
    else:
        test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using test device: {test_device}")
        print(f"Attempting GPU FBP_CUDA, will fallback to CPU FBP if needed.")

        # Test with full views first
        print("\n--- Testing with Full Sinogram ---")
        dummy_sino_batch_full = torch.rand(2, 1, config.NUM_ANGLES_FULL, config.DETECTORS_COUNT, device=test_device)
        print(f"Input full fanflat sinogram shape: {dummy_sino_batch_full.shape}")
        print(f"Using angles: 0 to {np.rad2deg(config.ANGLES_RAD[-1]):.1f} deg ({len(config.ANGLES_RAD)} views)")
        
        start_time = time.time()
        recon_full = fbp_reconstruct(dummy_sino_batch_full, config.ANGLES_RAD, config.IMAGE_SIZE)
        end_time = time.time()
        print(f"Reconstruction (full) finished. Shape: {recon_full.shape}")
        print(f"Total reconstruction time for full batch: {end_time - start_time:.4f}s")

        # Test with sparse views
        print("\n--- Testing with Sparse Sinogram ---")
    
        num_sparse_angles_for_test = len(config.SPARSE_ANGLES_RAD)
        dummy_sino_batch_sparse = torch.rand(2, 1, num_sparse_angles_for_test, config.DETECTORS_COUNT, device=test_device)

        print(f"Input sparse fanflat sinogram shape: {dummy_sino_batch_sparse.shape}")
        print(f"Using sparse angles: ({len(config.SPARSE_ANGLES_RAD)} views)")

        start_time = time.time()
        recon_sparse = fbp_reconstruct(dummy_sino_batch_sparse, config.SPARSE_ANGLES_RAD, config.IMAGE_SIZE)
        end_time = time.time()
        print(f"Reconstruction (sparse) finished. Shape: {recon_sparse.shape}")
        print(f"Total reconstruction time for sparse batch: {end_time - start_time:.4f}s")


        try:
            import matplotlib.pyplot as plt
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(dummy_sino_batch_full[0, 0].cpu().numpy(), cmap='gray', aspect='auto')
            axes[0, 0].set_title('Input Full Sino [0]')
            axes[0, 0].axis('off')
            axes[0, 1].imshow(recon_full[0, 0].cpu().numpy(), cmap='gray')
            axes[0, 1].set_title('ASTRA FBP from Full Sino [0]')
            axes[0, 1].axis('off')

            axes[1, 0].imshow(dummy_sino_batch_sparse[0, 0].cpu().numpy(), cmap='gray', aspect='auto')
            axes[1, 0].set_title(f'Input Sparse Sino [0]\n({num_sparse_angles_for_test} views)')
            axes[1, 0].axis('off')
            axes[1, 1].imshow(recon_sparse[0, 0].cpu().numpy(), cmap='gray')
            axes[1, 1].set_title('ASTRA FBP from Sparse Sino [0]')
            axes[1, 1].axis('off')

            save_path = os.path.join(config.OUTPUT_DIR, 'fbp_astra_fanflat_test_combined.png')
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Saved test reconstruction viz to {save_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Could not save test visualization: {e}")

    print("--- reconstruction.py test finished ---")

