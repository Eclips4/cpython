{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ (import ./python.nix { inherit pkgs; }) pkgs.z3 pkgs.ruff ];
}
