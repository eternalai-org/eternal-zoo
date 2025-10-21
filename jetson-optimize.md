# Power & system prep (host)

```bash
# 1) Max performance & lock clocks
sudo nvpmodel -m 0            # MAXN
sudo jetson_clocks

# 2) Give CUDA a big JIT cache (persists across runs)
echo 'export CUDA_CACHE_MAXSIZE=2147483647' | sudo tee -a /etc/profile.d/cuda_cache.sh
echo 'export CUDA_CACHE_PATH=/var/cache/cuda' | sudo tee -a /etc/profile.d/cuda_cache.sh
sudo mkdir -p /var/cache/cuda && sudo chmod 777 /var/cache/cuda

# 3) Optional: ZRAM swap to prevent OOM while keeping perf OK
# (JetPack 6 usually has zram-on by default; verify or set to ~8–16 GB)
cat /proc/swaps
```

## System-wide (works for all shells & services)

```bash
# Put plain KEY=VALUE lines (no 'export') in /etc/environment
echo 'CUDA_CACHE_MAXSIZE=2147483647' | sudo tee -a /etc/environment
echo 'CUDA_CACHE_PATH=/var/cache/cuda' | sudo tee -a /etc/environment
# Re-login (or `sudo loginctl terminate-user $USER`) for it to take effect.
```
