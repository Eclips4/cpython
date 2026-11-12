{ pkgs ? import <nixpkgs> { } }:
pkgs.python3.withPackages (ps: [
  ps.z3-solver
  ps.mypy
  (ps.buildPythonPackage rec {
    pname = "adaptix";
    version = "3.0.0b12";
    format = "wheel";
    src = ps.fetchPypi {
      inherit pname version format;
      dist = "py3";
      python = "py3";
      hash = "sha256-gVbwVJuW/NGT0T78p7+9o5s9Y4kIVVbZMOIEwJ+a5u0=";
    };
  })
])
