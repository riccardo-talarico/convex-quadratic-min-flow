from dataclasses import dataclass
import numpy as np

# min{ x^T Q x +qx: Ex=b, 0 \leq x \leq u}
# E = n x m node-arc incidence matrix, n nodes and m edges

@dataclass
class ConvexMinFlow():
    Q : np.array # Since it is diagonal I can accept an array with size (m,)
    q : np.array
    E : np.array
    b : np.array
    u : np.array

    def check_init(self) -> bool:
        """
        The function checks if the data is a well-formed problem instance.\\
        It checks if the shapes match, checks if Q is positive semidefinite and turns it into an array if it is a matrix.
        
        It prints an error message for the first problem it finds.

        Returns:
            bool: True if it is a well-formed instance, False otherwise.
        """
        try:
            # Checking correct shapes
            (n,m) = self.E.shape
            if self.b.shape != (n,) or self.u.shape != (m,) or self.q.shape != (m,):
                raise ValueError("wrong dimensions.")
            
            # Checking Q shape and converting to a vector if it is a matrix
            if len(self.Q.shape) == 2:
                if self.Q.shape != (m,m):
                    raise ValueError(f"Wrong Q dimensions, expected {(m,m)}, got: {self.Q.shape}")
                
                # Converting into a vector:
                d = np.diag(self.Q)
                # Checking if it is positive semidefinite
                if any(d < 0):
                    raise ValueError(f"Expected Q positive semidefinite, got {self.Q}")
                # Checking if it is actually diagonal
                if any(self.Q - d != 0):
                    raise ValueError(f"Expected Q diagonal, got {Q}")
                # Storing just the diagonal
                self.Q = d
            elif len(self.Q.shape) == 1 and self.Q.shape == (m,):
                # Checking that Q is positive semidefinite
                if any(self.Q < 0):
                    raise ValueError(f"Expected Q positive semidefinite, got {self.Q}")
            else:
                raise ValueError(f"Wrong Q shape, expected {(m,)} or {(m,m)}, got {self.Q.shape}")
                
            # Checking positivity of u
            if any(self.u < 0):
                raise ValueError("vector u must be positive.")
            
        except Exception as e:
            print(f"MinFlow instance is not well formed: {e}")
            return False
        return True
    

if __name__ == '__main__':
    cmf = ConvexMinFlow(np.array([1,-1,1]),np.array([0,0,1]),np.array([[1,0,0],[0,1,0],[0,0,1]]),np.array([1,0,0]),np.array([1,1,1]))
    assert cmf.check_init() == False
    cmf.Q = np.array([1,1])
    assert cmf.check_init() == False
    cmf.Q = np.array([1,1,1])
    assert cmf.check_init()


