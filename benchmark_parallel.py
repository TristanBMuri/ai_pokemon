import subprocess
import time
import re
import matplotlib.pyplot as plt

def benchmark_parallel(max_envs=64, steps_per_run=500):
    results = []
    
    # Test powers of 2 up to max_envs
    n_envs_list = [1]
    curr = 2
    while curr <= max_envs:
        n_envs_list.append(curr)
        curr *= 2
        
    print(f"Benchmarking n_envs: {n_envs_list}")
    
    for n_envs in n_envs_list:
        print(f"\nTesting n_envs={n_envs}...", flush=True)
        start_time = time.time()
        
        # Run training command
        # We use a small number of steps just to get it running and stabilize
        cmd = [
            "uv", "run", "python", "-u", "train_manager.py",
            "--steps", str(steps_per_run),
            "--n_envs", str(n_envs),
            "--mock" # Use mock for pure throughput test, or remove for real
        ]
        
        # Note: Using --mock to test overhead of SubprocVecEnv vs DummyVecEnv
        # If user wants to test Showdown capacity, we should remove --mock.
        # The user asked "test the maximum number of parallel environments", likely implying Showdown capacity.
        # So let's default to REAL (remove --mock), but maybe lower steps.
        
        cmd = [
            "uv", "run", "python", "-u", "train_manager.py",
            "--steps", str(steps_per_run),
            "--n_envs", str(n_envs),
            "--n_steps_per_update", "64" # Force frequent updates
        ]
        
        try:
            # Capture output to find FPS
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            fps_values = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    # SB3 output: "fps | 123"
                    # Or "fps             | 123"
                    if "fps" in line:
                        match = re.search(r"fps\s+\|\s+(\d+)", line)
                        if match:
                            fps = int(match.group(1))
                            fps_values.append(fps)
                            print(f"  FPS: {fps}", flush=True)
                            
            process.wait()
            
            if process.returncode != 0:
                print(f"  Failed with exit code {process.returncode}")
                avg_fps = 0
            else:
                if fps_values:
                    avg_fps = sum(fps_values) / len(fps_values)
                else:
                    # Fallback if no FPS log found (short run)
                    duration = time.time() - start_time
                    avg_fps = steps_per_run / duration
                    
            print(f"  Average FPS: {avg_fps:.1f}")
            results.append((n_envs, avg_fps))
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append((n_envs, 0))
            
    print("\n--- Benchmark Results ---")
    print(f"{'n_envs':<10} | {'FPS':<10}")
    print("-" * 25)
    for n, fps in results:
        print(f"{n:<10} | {fps:<10.1f}")
        
    # Plot
    try:
        ns, fpss = zip(*results)
        plt.figure()
        plt.plot(ns, fpss, marker='o')
        plt.xlabel('Number of Environments')
        plt.ylabel('FPS')
        plt.title('Parallel Training Performance')
        plt.grid(True)
        plt.savefig('benchmark_plot.png')
        print("\nPlot saved to benchmark_plot.png")
    except:
        pass

if __name__ == "__main__":
    # Test up to 64 environments
    benchmark_parallel(max_envs=64, steps_per_run=128)
