{
  description = "Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }@inputs:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python3 = pkgs.python3.override {
          packageOverrides = python-self: python-super: {
            jsonargparse = python-super.buildPythonPackage rec {
              pname = "jsonargparse";
              version = "4.18.0";
              src = python-super.fetchPypi {
                inherit pname version;
                sha256 = "sha256-LnuINC5lkWallRcqRic0lX2dSrXqsAapqusmrlxkZSU=";
              };

              # Ensure pip is available during the build process
              nativeBuildInputs = with python-super; [ pip setuptools wheel ];

              # Include runtime dependencies
              propagatedBuildInputs = with python-super; [ setuptools pyyaml ];

              # This line is essential for some packages that use setup.py
              # It ensures the build process uses PEP 517/518
              format = "pyproject";
            };
            docstring-parser = python-super.buildPythonPackage rec {
              pname = "docstring_parser";
              version = "0.15";
              src = python-super.fetchPypi {
                inherit pname version;
                sha256 = "sha256-SN3Ak+ixhliZlW/MA7A+ZrtyQMMQ+sWvgYFFgMVb9oI=";
              };

              nativeBuildInputs = with python-super; [ pytest ];

              propagatedBuildInputs = with python-super; [ setuptools ];
            };
          };

        };

        pythonEnv = python3.withPackages (ps:
          with ps; [
            matplotlib
            opencv4
            pip
            pillow
            tqdm
            mercantile
            rasterio
            pandas
            pytest
            requests
            pytorch-lightning
            gdal
            timm
            wandb
            configargparse
            jsonargparse
            docstring-parser
            black
          ]);
      in {
        devShells.default = pkgs.mkShell { buildInputs = [ pythonEnv pkgs.yamlfmt ]; };
      });

}
