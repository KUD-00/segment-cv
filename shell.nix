{ pkgs ? import <nixpkgs>{} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      ps.moviepy
      ps.imageio
      ps.setuptools
      ps.pillow
      ps.opencv4
    ]))

    pkgs.curl
    pkgs.jq
  ];
}
