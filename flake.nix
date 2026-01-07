{
  description = "Python Template";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs = { self , nixpkgs ,... }:
  let
    system = "x86_64-linux";
  in {
    devShells."${system}".default =
    let
      pkgs = import nixpkgs {
        inherit system;
      };
      
    in pkgs.mkShell {
      packages = with pkgs; [
        python312Packages.python
        python312Packages.jupyter
        python312Packages.notebook
        python312Packages.pip
        python312Packages.scipy
        python312Packages.pandas
        python312Packages.numpy
        python312Packages.torch
        python312Packages.matplotlib
        python312Packages.pandas
        python312Packages.rpy2
      ];
    };
  };
}
