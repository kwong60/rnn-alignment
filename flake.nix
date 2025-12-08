{
  description = "RNN Alignment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            uv # Fast dependency resolver and wheel installer
          ];

          shellHook = ''
            echo "🚀 Entering fast Python environment (uv + wheels)"
            echo "Python version: $(python3 --version)"
            echo ""
            uv sync
            source .venv/bin/activate
            echo "Virtualenv activated (packages installed with uv)"
          '';
        };
      });
}

