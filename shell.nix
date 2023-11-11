{ pkgs ? import <nixpkgs>{} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      ps.moviepy
      ps.imageio
      ps.setuptools
      ps.pillow
    ]))

    pkgs.curl
    pkgs.jq
  ];
}
