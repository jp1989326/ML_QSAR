from molml.utils import get_coulomb_matrix
from molml.features import CoulombMatrix, BagOfBonds
import numpy 

class Cust_cm(CoulombMatrix):
    def __init__(self,input_type='list', n_jobs=1, sort=False, eigen=False):
        super(CoulombMatrix, self).__init__()
        self._max_size = None
        self.sort = sort
        self.eigen = eigen    
        
    def _para_transform(self, X):
        """
        A single instance of the transform procedure.

        This is formulated in a way that the transformations can be done
        completely parallel with map.

        Parameters
        ----------
        X : object
            An object to use for the transform

        Returns
        -------
        value : array
            The features extracted from the molecule

        Raises
        ------
        ValueError
            If the transformer has not been fit.

        ValueError
            If the size of the transforming molecules are larger than the fit.
        """
        if self._max_size is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        data = self.convert_input(X)
        if len(data.numbers) > self._max_size:
            msg = "The fit molecules (%d) were not as large as the ones that"
            msg += " are being transformed (%d)."
            raise ValueError(msg % (self._max_size, len(data.numbers)))

        padding_difference = self._max_size - len(data.numbers)
        values = get_coulomb_matrix(data.numbers, data.coords)
        if self.sort:
            order = numpy.argsort(values.sum(0))[::-1]
            values = values[order, :][:, order]

        if self.eigen:
            values = numpy.linalg.eig(values)[0]

        values = numpy.pad(values,
                           (0, padding_difference),
                           mode="constant")
        return values
#        return values.reshape(-1)
