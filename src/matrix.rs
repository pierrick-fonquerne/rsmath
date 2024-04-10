// rsmath/src/matrix.rs

use std::{
    fmt::{Debug, Formatter},
    ops::{Index, IndexMut},
};

/// A matrix of floating point numbers.
#[derive(Clone)]
pub struct Matrix {
    /// The matrix data, stored in row-major order.
    data: Vec<f64>,
    /// The number of rows in the matrix.
    rows: usize,
    /// The number of columns in the matrix.
    cols: usize,
}

impl Matrix {
    /// Swaps two rows of the matrix.
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the first row to swap.
    /// * `j` - The index of the second row to swap.
    fn swap_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.cols {
            let index1 = self.index(i, k).unwrap();
            let index2 = self.index(j, k).unwrap();
            self.data.swap(index1, index2);
        }
    }

    /// Returns a reference to the underlying data vector.
    pub fn as_vec(&self) -> &[f64] {
        &self.data
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Creates a new matrix with the given number of rows and columns, with all values initialized to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let matrix = Matrix::new(2, 3).unwrap();
    ///
    /// for row in 0..matrix.rows() {
    ///     for col in 0..matrix.cols() {
    ///         assert_eq!(matrix.get(row, col), Ok(0.0));
    ///     }
    /// }
    /// ```
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

    /// Creates a new matrix from the specified data.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows in the matrix.
    /// * `cols` - The number of columns in the matrix.
    /// * `data` - The matrix data, in row-major order.
    ///
    /// # Returns
    ///
    /// A new `Matrix` instance, or an error if the length of `data` does not match the specified dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let matrix = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert_eq!(matrix.get(0, 0), Ok(1.0));
    /// assert_eq!(matrix.get(0, 1), Ok(2.0));
    /// assert_eq!(matrix.get(0, 2), Ok(3.0));
    /// assert_eq!(matrix.get(1, 0), Ok(4.0));
    /// assert_eq!(matrix.get(1, 1), Ok(5.0));
    /// assert_eq!(matrix.get(1, 2), Ok(6.0));
    /// ```
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, &'static str> {
        if data.len() != rows * cols {
            return Err("Data length does not match matrix dimensions");
        }

        Ok(Matrix { data, rows, cols })
    }

    /// Returns the index of the specified element in the matrix data vector.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    ///
    /// # Returns
    ///
    /// The index of the element in the matrix data vector, or an error if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let matrix = Matrix::new(2, 3).unwrap();
    /// assert_eq!(matrix.index(0, 0), Ok(0));
    /// assert_eq!(matrix.index(1, 2), Ok(5));
    /// ```
    pub fn index(&self, row: usize, col: usize) -> Result<usize, &'static str> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds");
        }

        Ok(row * self.cols + col)
    }

    /// Returns the value of the specified element in the matrix.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    ///
    /// # Returns
    ///
    /// The value of the element, or an error if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let matrix = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// assert_eq!(matrix.get(0, 0), Ok(1.0));
    /// assert_eq!(matrix.get(1, 2), Ok(6.0));
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Result<f64, &'static str> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds");
        }

        let index = row * self.cols + col;
        Ok(self.data[index])
    }

    /// Sets the value of the specified element in the matrix.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index of the element.
    /// * `col` - The column index of the element.
    /// * `value` - The new value of the element.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the value was set successfully, or an error if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let mut matrix = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// matrix.set(0, 0, 7.0).unwrap();
    /// assert_eq!(matrix.get(0, 0), Ok(7.0));
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), &'static str> {
        let index = self.index(row, col)?;
        self.data[index] = value;
        Ok(())
    }

    /// Performs LU decomposition on a matrix.
    ///
    /// # Arguments
    ///
    /// * `mat` - The matrix to decompose.
    ///
    /// # Returns
    ///
    /// A tuple containing the LU decomposed matrix and a vector of pivot indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    /// let (lu, pivots) = Matrix::lu_decomposition(&mat).unwrap();
    /// assert_eq!(lu.data, vec![1.0, 2.0, 3.0, 4.0, 13.0 / 2.0, 1.0, 7.0, 20.0 / 3.0, 1.0]);
    /// assert_eq!(pivots, vec![0, 1, 2]);
    /// ```
    pub fn lu_decomposition(mat: &Matrix) -> Result<(Matrix, Vec<usize>), &'static str> {
        let mut a = mat.clone();
        let mut pivots = vec![0; a.rows];

        for i in 0..a.rows {
            // Find pivot row with maximum absolute value
            let mut max_pivot = 0.0;
            let mut max_index = 0;
            for j in i..a.rows {
                let abs_pivot = a[(j, i)].abs();
                if abs_pivot > max_pivot {
                    max_pivot = abs_pivot;
                    max_index = j;
                }
            }

            // Check for singular matrix
            if max_pivot == 0.0 {
                return Err("Matrix is singular");
            }

            // Record pivot index and swap rows
            pivots[i] = max_index;
            a.swap_rows(i, max_index);

            // Perform elimination step
            for j in i + 1..a.rows {
                let factor = a[(j, i)] / a[(i, i)];
                a.set(j, i, factor)?;
                for k in i + 1..a.cols {
                    a.set(j, k, a[(j, k)] - factor * a[(i, k)])?;
                }
            }
        }

        Ok((a, pivots))
    }

    /// Performs back substitution on a matrix.
    ///
    /// # Arguments
    ///
    /// * `pivots` - The pivot indices from the LU decomposition.
    /// * `b` - The matrix to perform back substitution on.
    ///
    /// # Returns
    ///
    /// The resulting matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let mat = Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    /// let (lu, pivots) = Matrix::lu_decomposition(&mat).unwrap();
    /// let b = Matrix::from_data(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    /// let x = lu.back_substitution(&pivots, b).unwrap();
    /// assert_eq!(x.as_vec().as_slice(), vec![-3.0, 2.0, -1.0]);
    /// ```
    pub fn back_substitution(&self, pivots: &[usize], b: Matrix) -> Result<Matrix, &'static str> {
        let mut x = b;

        // Forward substitution
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..i {
                sum += x[(pivots[i], j)] * x[(j, i)];
            }
            x.set(pivots[i], i, x[(pivots[i], i)] - sum)?;
        }

        // Backward substitution
        for i in (0..self.rows).rev() {
            let mut sum = 0.0;
            for j in i + 1..self.rows {
                sum += x[(pivots[i], j)] * x[(j, i)];
            }
            x.set(pivots[i], i, (x[(pivots[i], i)] - sum) / x[(i, i)])?;
        }

        Ok(x)
    }

    /// Creates an identity matrix of the specified size.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the identity matrix.
    ///
    /// # Returns
    ///
    /// The identity matrix, or an error if the size is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let matrix = Matrix::identity(3).unwrap();
    /// let expected = Matrix::from_data(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    /// assert_eq!(matrix, expected);
    /// ```
    pub fn identity(size: usize) -> Result<Matrix, &'static str> {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Matrix::from_data(size, size, data)
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

    pub fn inverse(&self) -> Result<Matrix, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not square");
        }
        let (lu, pivots) = Self::lu_decomposition(self)?;
        let inv = lu.back_substitution(&pivots, Matrix::identity(self.rows)?)?;
        Ok(inv)
    }

    pub fn determinant(&self) -> Result<f64, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not square");
        }
        let (lu, pivots) = Self::lu_decomposition(self)?;
        let mut det = 1.0;
        for i in 0..self.rows {
            det *= lu[(i, i)];
            if pivots[i] != i {
                det *= -1.0;
            }
        }
        Ok(det)
    }

    pub fn solve(&self, b: &Matrix) -> Result<Matrix, &'static str> {
        if self.rows != self.cols || b.rows != self.rows || b.cols != 1 {
            return Err("Matrix dimensions do not match");
        }
        let (lu, pivots) = Self::lu_decomposition(self)?;
        let x = lu.back_substitution(&pivots, b.clone())?;
        Ok(x)
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

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
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
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);

        let mat3 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mat4 = Matrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = mat3.mul(&mat4).unwrap();
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_transpose() {
        let mat = Matrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = mat.transpose();
        let expected = Matrix::from_data(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        assert_eq!(result, expected);

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let result = mat.transpose();
        let expected =
            Matrix::from_data(3, 3, vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_inverse() {
        let mat = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = mat.inverse().unwrap();
        let expected = Matrix::from_data(2, 2, vec![-2.0, 1.0, 1.5, -0.5]).unwrap();
        assert_eq!(result, expected);

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let result = mat.inverse();
        assert!(result.is_err());

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0]).unwrap();
        let result = mat.inverse().unwrap();
        let expected = Matrix::from_data(
            3,
            3,
            vec![
                -3.0,
                1.5,
                0.5,
                1.5,
                -0.5,
                0.0,
                0.5,
                0.0,
                -0.16666666666666666,
            ],
        )
        .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_determinant() {
        let mat = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = mat.determinant().unwrap();
        let expected = -2.0;
        assert_eq!(result, expected);

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let result = mat.determinant().unwrap();
        let expected = 0.0;
        assert_eq!(result, expected);

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0]).unwrap();
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

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let b = Matrix::from_data(3, 1, vec![10.0, 11.0, 12.0]).unwrap();
        let result = mat.solve(&b);
        assert!(result.is_err());

        let mat =
            Matrix::from_data(3, 3, vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0]).unwrap();
        let b = Matrix::from_data(3, 1, vec![10.0, 11.0, 12.0]).unwrap();
        let result = mat.solve(&b).unwrap();
        let expected = Matrix::from_data(3, 1, vec![1.0, -1.0, 1.0]).unwrap();
        assert_eq!(result, expected);
    }
}
