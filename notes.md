python -m pytest tests/

On a un fichier geof tel que :

<!-- NODES -->
0 | x0 y0 z0
...
N | xN yN zN
<!-- CELLS -->
0 | a  b  c
...
M | l m n

On a :

1D cell vertices = [[0.1], [1.1]]
2D cell vertices = [[0.0, 1.2], [2.3, 8.9], [3.8, 7.4]]
2D cell vertices = [[0.0, 1.2, 2.9], [2.3, 8.9, 6.5], [3.8, 7.4, 8.5]]

6666666666666666
3333333333333333
2000000000000000
6000000000000000
5625000000000000
5208333333333333