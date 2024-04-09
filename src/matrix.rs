use std::simd::{f64x2, f64x4};

impl Matrix {
    /// Add two matrices together.
    ///
    /// # Arguments
    ///
    /// * `other` - The matrix to add to `self`.
    ///
    /// # Returns
    ///
    /// A new matrix representing the sum of `self` and `other`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions of `self` and `other` do not match.
    ///
    /// # Examples
    ///
    /// ```
    /// use rsmath::Matrix;
    ///
    /// let mat1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let mat2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    /// let result = mat1.add(&mat2).unwrap();
    /// assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    /// ```
    pub fn add(&self, other: &Matrix) -> Result<Self, &'static str> {
        // Check that the dimensions of `self` and `other` match.
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions do not match");
        }

        // Create a new matrix with the same dimensions as `self`.
        let mut result = Matrix::new(self.rows, self.cols)?;

        // Use SIMD instructions to add the elements of `self` and `other`.
        let mut i = 0;
        while i + 3 < self.rows * self.cols {
            // Load four elements from `self` and `other` into SIMD registers.
            let a = f64x4::loadu(&self.data[i..]);
            let b = f64x4::loadu(&other.data[i..]);

            // Add the elements of `a` and `b` using SIMD instructions.
            let c = a + b;

            // Store the result in `result`.
            c.store(&mut result.data[i..]);

            // Increment the index by 4.
            i += 4;
        }

        // Add any remaining elements using SIMD instructions.
        while i + 1 < self.rows * self.cols {
            // Load two elements from `self` and `other` into SIMD registers.
            let a = f64x2::loadu(&self.data[i..]);
            let b = f64x2::loadu(&other.data[i..]);

            // Add the elements of `a` and `b` using SIMD instructions.
            let c = a + b;

            // Store the result in `result`.
            c.store(&mut result.data[i..]);

            // Increment the index by 2.
            i += 2;
        }

        // Add any remaining element using scalar instructions.
        if i < self.rows * self.cols {
            result.data[i] = self.data[i] + other.data[i];
        }

        Ok(result)
    }
}
