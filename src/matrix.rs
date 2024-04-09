// rsmath/src/matrix.rs

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
        let index = self.index(row, col)?;
        Ok(self.data[index])
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<(), &'static str> {
        let index = self.index(row, col)?;
        self.data[index] = value;
        Ok(())
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
    fn test_add_error() {
        let mat1 = Matrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mat2 = Matrix::from_data(2, 3, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        let result = mat1.add(&mat2);
        assert!(result.is_err());
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
}
