# Makefile for EternalZoo installation
# Sets specific versions for mlx-flux and eternalzoo packages

.PHONY: download

# Default Filecoin/IPFS hash (change as needed)
HASH=bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm

# Download model using the specified hash
# Usage:
#   make download            # uses the default HASH
#   make download HASH=your_filecoin_hash_here

download:
	python eternal_zoo/download.py $(HASH)

MLX_OPENAI_SERVER_TAG=1.3.4
ETERNAL_ZOO_TAG=2.0.34

# Default target
.PHONY: install
install-macos:
	@echo "Installing EternalZoo with specific versions:"
	@echo "  MLX_FLUX_TAG: $(MLX_FLUX_TAG)"
	@echo "  MLX_OPENAI_SERVER_TAG: $(MLX_OPENAI_SERVER_TAG)"
	@echo "  ETERNAL_ZOO_TAG: $(ETERNAL_ZOO_TAG)"
	@echo ""
	MLX_FLUX_TAG=$(MLX_FLUX_TAG) MLX_OPENAI_SERVER_TAG=$(MLX_OPENAI_SERVER_TAG) ETERNAL_ZOO_TAG=$(ETERNAL_ZOO_TAG) bash mac.sh

# Install target for Jetson
.PHONY: install-jetson
install-jetson:
	@echo "Installing EternalZoo for Jetson:"
	@echo "  ETERNAL_ZOO_TAG: $(ETERNAL_ZOO_TAG)"
	@echo ""
	ETERNAL_ZOO_TAG=$(ETERNAL_ZOO_TAG) bash jetson.sh

# Clean target to remove virtual environment
.PHONY: clean
clean:
	@echo "Cleaning up virtual environment..."
	rm -rf ~/.eternal-zoo/venv
	@echo "Cleanup complete."

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install  - Install EternalZoo with specific package versions"
	@echo "  clean    - Remove the virtual environment"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Package versions:"
	@echo "  MLX_FLUX_TAG: $(MLX_FLUX_TAG)"
	@echo "  MLX_OPENAI_SERVER_TAG: $(MLX_OPENAI_SERVER_TAG)"
	@echo "  ETERNALZOO_TAG: $(ETERNAL_ZOO_TAG)" 