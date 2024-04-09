# RsMath
RsMath is a simple mathematical library for Rust, designed for AI, ML, and deep learning projects.

## Features
- Basic matrix operations: addition, subtraction, multiplication, transposition, etc.
- Basic vector operations: addition, subtraction, dot product, norm, etc.
- Common activation functions: sigmoid, ReLU, tanh, etc.
- Common loss functions: mean squared error, cross-entropy, etc.
- Common optimizers: stochastic gradient descent, Adam, RMSProp, etc.

## Installation
To use RsMath in your Rust project, add the following dependency to your Cargo.toml file:

```
[dependencies]
rsmath = "0.1.0"
```

## Usage
Here's an example of using RsMath to perform matrix multiplication:

```
use rsmath::matrix::Matrix;

fn main() {
    let mat1 = Matrix::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
    let mat2 = Matrix::new(vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]);
    let result = mat1.mm(&mat2);
    println!("{}", result);
}
```

This should print the resulting matrix:
```
19.0 22.0
43.0 50.0
```

## Contribution
Contributions to RsMath are welcome! If you would like to contribute, please read our contribution guide for more information.

## License
RsMath is licensed under the Creative Commons Universal (CC0 1.0) license. See the LICENSE file for more information.

------
# RsMath
RsMath est une bibliothèque mathématique simple pour Rust, conçue pour les projets d'IA, de ML et d'apprentissage profond.

## Fonctionnalités
- Opérations matricielles de base : addition, soustraction, multiplication, transposition, etc.
- Opérations vectorielles de base : addition, soustraction, produit scalaire, norme, etc.
- Fonctions d'activation courantes : sigmoïde, ReLU, tanh, etc.
- Fonctions de perte courantes : carré moyen, entropie croisée, etc.
- Optimiseurs courants : descente de gradient stochastique, Adam, RMSProp, etc.

## Installation
Pour utiliser RsMath dans votre projet Rust, ajoutez la dépendance suivante à votre fichier Cargo.toml :

```
[dependencies]
rsmath = "0.1.0"
```

Notez que la version actuelle de RsMath est 0.1.0. Vérifiez la dernière version sur la page de publication de GitHub.

## Utilisation
Voici un exemple d'utilisation de RsMath pour effectuer une multiplication matricielle :

```
use rsmath::matrix::Matrix;

fn main() {
    let mat1 = Matrix::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
    let mat2 = Matrix::new(vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]);
    let result = mat1.mm(&mat2);
    println!("{}", result);
}
```

Cela devrait afficher la matrice résultante :

```
19.0 22.0
43.0 50.0
```

## Contribution
Les contributions à RsMath sont les bienvenues ! Si vous souhaitez contribuer, veuillez lire notre guide de contribution pour plus d'informations.

## Licence
RsMath est sous licence Creative Commons Universal (CC0 1.0). Consultez le fichier LICENCE pour plus d'informations.

