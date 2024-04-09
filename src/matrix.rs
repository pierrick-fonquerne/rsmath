// rsmath/src/matrix.rs

use std::fmt::Debug;
use std::fmt::Formatter;

#[derive(Clone)]
pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Result<Self, &'static str> {
        if rows == 0 || cols == 0 {
            return Err("Matrix dimensions cannot be zero");
        }

        Ok(Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        })
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, &'static str> {
        if data.len() != rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        Ok(Matrix { data, rows, cols })
    }

    pub fn index(&self, row: usize, col: usize) -> Result<usize, &'static str> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds");
        }

        Ok(row * self.cols + col)
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f64, &'static str> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds");
        }

        let index = row * self.cols + col;
        Ok(self.data[index])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), &'static str> {
        let index = self.index(row, col)?;
        self.data[index] = value;
        Ok(())
    }

    fn swap_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.cols {
            let index1 = self.index(i, k).unwrap();
            let index2 = self.index(j, k).unwrap();
            self.data.swap(index1, index2);
        }
    }

    pub fn add(&self, other: &Matrix) -> Result<Self, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions do not match");
        }

        let mut result = Matrix::new(self.rows, self.cols)?;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = result.index(i, j)?;
                result.data[index] = self.get(i, j)? + other.get(i, j)?;
            }
        }

        Ok(result)
    }

    pub fn sub(&self, other: &Matrix) -> Result<Self, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions do not match");
        }

        let mut result = Matrix::new(self.rows, self.cols)?;

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = result.index(i, j)?;
                result.data[index] = self.get(i, j)? - other.get(i, j)?;
            }
        }

        Ok(result)
    }

    pub fn mul(&self, other: &Matrix) -> Result<Self, &'static str> {
        if self.cols != other.rows {
            return Err("Matrix dimensions do not match");
        }

        let mut result = Matrix::new(self.rows, other.cols)?;

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k)? * other.get(k, j)?;
                }
                let index = result.index(i, j)?;
                result.data[index] = sum;
            }
        }

        Ok(result)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Matrix::new(self.cols, self.rows).unwrap();

        for i in 0..self.rows {
            for j in 0..self.cols {
                let index = result.index(j, i).unwrap();
                result.data[index] = self.get(i, j).unwrap();
            }
        }

        result
    }

    pub fn inverse(&self) -> Result<Self, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not square");
        }

        let det = self.determinant()?;
        if det == 0.0 {
            return Err("Matrix is singular");
        }

        let mut result = Matrix::new(self.rows, self.cols)?;
        let mut temp = self.clone();

        for i in 0..self.rows {
            if temp.get(i, i)? == 0.0 {
                for j in i+1..self.rows {
                    if temp.get(j, i)? != 0.0 {
                        temp.swap_rows(i, j);
                        result.swap_rows(i, j);
                        break;
                    }
                }
            }

            let pivot = temp.get(i, i)?;
            for j in i..self.cols {
                temp.set(i, j, temp.get(i, j)? / pivot)?;
                result.set(i, j, result.get(i, j)? / pivot)?;
            }

            for j in 0..self.rows {
                if j == i {
                    continue;
                }

                let factor = temp.get(j, i)?;
                for k in i..self.cols {
                    temp.set(j, k, temp.get(j, k)? - factor * temp.get(i, k)?)?;
                    result.set(j, k, result.get(j, k)? - factor * result.get(i, k)?)?;
                }
            }
        }

        for i in (0..self.rows).rev() {
            for j in 0..i {
                let factor = temp.get(j, i)?;
                for k in 0..self.cols {
                    temp.set(j, k, temp.get(j, k)? - factor * temp.get(i, k)?)?;
                    result.set(j, k, result.get(j, k)? - factor * result.get(i, k)?)?;
                }
            }
        }

        Ok(result)
    }

    pub fn determinant(&self) -> Result<f64, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not square");
        }

        let mut temp = self.clone();
        let mut det = 1.0;
        let mut sign = 1.0;

        for i in 0..self.rows {
            if temp.get(i, i)? == 0.0 {
                for j in i+1..self.rows {
                    if temp.get(j, i)? != 0.0 {
                        temp.swap_rows(i, j);
                        sign = -sign;
                        break;
                    }
                }
            }

            let pivot = temp.get(i, i)?;
            det *= pivot;
            for j in i+1..self.cols {
                temp.set(i, j, temp.get(i, j)? / pivot)?;
            }

            for j in i+1..self.rows {
                let factor = temp.get(j, i)?;
                for k in i+1..self.cols {
                    temp.set(j, k, temp.get(j, k)? - factor * temp.get(i, k)?)?;
                }
            }
        }

        Ok(sign * det)
    }

    pub fn solve(&self, b: &Matrix) -> Result<Matrix, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not square");
        }

        if b.rows != self.rows {
            return Err("Matrix dimensions do not match");
        }

        let inv = self.inverse()?;
        Ok(inv.mul(b)?)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.get(i, j).unwrap() != other.get(i, j).unwrap() {
                    return false;
                }
            }
        }

        true
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix({}x{}) {{ ", self.rows, self.cols)?;

        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{} ", self.get(i, j).unwrap())?;
            }
            writeln!(f)?;
        }

        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let mat1 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mat2 = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = mat1.add(&mat2).unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);

        let mat3 = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mat4 = Matrix::from_data(2, 3, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]).unwrap();
        let result = mat3.add(&mat4).unwrap();
        assert_eq!(result.data, vec![7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
    }

    #[test]
    fn test_sub() {
        let mat1 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mat2 = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = mat1.sub(&mat2).unwrap();
        assert_eq!(result.data, vec![-4.0, -4.0, -4.0, -4.0]);

        let mat3 = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mat4 = Matrix::from_data(2, 3, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]).unwrap();
        let result = mat3.sub(&mat4).unwrap();
        assert_eq!(result.data, vec![-5.0, -5.0, -5.0, -5.0, -5.0, -5.0]);
    }

    #[test]
    fn test_mul() {
        let mat1 = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mat2 = Matrix::from_data(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = mat1.mul(&mat2).unwrap();
        assert_eq!(result.data, vec![58.0, 139.0, 116.0, 299.0]);

        let mat3 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mat4 = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = mat3.mul(&mat4).unwrap();
        assert_eq!(result.data, vec![19.0, 43.0, 43.0, 99.0]);
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = mat.transpose();
        let expected = Matrix::from_data(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        assert_eq!(result, expected);

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let result = mat.transpose();
        let expected = Matrix::from_data(3, 3, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse() {
        let mat = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = mat.inverse().unwrap();
        let expected = Matrix::from_data(2, 2, vec![-2.0, 1.0, 1.5, -0.5]).unwrap();
        assert_eq!(result, expected);

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let result = mat.inverse();
        assert!(result.is_err());

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0]).unwrap();
        let result = mat.inverse().unwrap();
        let expected = Matrix::from_data(3, 3, vec![-3.0, 1.5, 0.5, 1.5, -0.5, 0.0, 0.5, 0.0, -0.16666666666666666]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_determinant() {
        let mat = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = mat.determinant().unwrap();
        let expected = -2.0;
        assert_eq!(result, expected);

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let result = mat.determinant().unwrap();
        let expected = 0.0;
        assert_eq!(result, expected);

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0]).unwrap();
        let result = mat.determinant().unwrap();
        let expected = -6.0;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_solve() {
        let mat = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_data(2, 1, vec![5.0, 6.0]).unwrap();
        let result = mat.solve(&b).unwrap();
        let expected = Matrix::from_data(2, 1, vec![-7.0, 2.5]).unwrap();
        assert_eq!(result, expected);

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let b = Matrix::from_data(3, 1, vec![10.0, 11.0, 12.0]).unwrap();
        let result = mat.solve(&b);
        assert!(result.is_err());

        let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0]).unwrap();
        let b = Matrix::from_data(3, 1, vec![10.0, 11.0, 12.0]).unwrap();
        let result = mat.solve(&b).unwrap();
        let expected = Matrix::from_data(3, 1, vec![1.0, -1.0, 1.0]).unwrap();
        assert_eq!(result, expected);
    }
}