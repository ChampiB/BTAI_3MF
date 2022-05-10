import torch


class Operators:

    @staticmethod
    def expansion(x1, n, dim):
        """
        Expand the input tensor along a dimension by repating its content n times.
        :param x1: the input tensor.
        :param n: the number of times the content needs to be repeated.
        :param dim: the dimension along with the tensor must be expanded.
        :return: the expanded tensor.
        """
        result = torch.unsqueeze(x1, dim)
        return result.repeat([n if i == dim else 1 for i in range(result.dim())])

    @staticmethod
    def multiplication(x1, x2, ml):
        """
        Multiply two tensors of potentially different shape element-wise. x2 must have fewer
        or the same number of dimensions than x2. x2 will be expanded until it has the same
        number of dimensions than x2.
        :param x1: the first tensor to multiply.
        :param x2: the second tensor to multiply.
        :param ml: a list describing how the dimensions of the x1 matches the dimesions of x2.
        :return: the result of the element-wise multiplication.
        """
        # Create the list on non-matching dimensions
        not_ml = []
        for i in range(x1.dim()):
            if i not in ml:
                not_ml.append(i)

        # Sequence of expansions
        x2_tmp = x2
        for i in not_ml:
            x2_tmp = Operators.expansion(x2_tmp, x1.shape[i], x2_tmp.dim())

        # Permutation
        pl = [0] * x1.ndim
        for i in range(x1.ndim):
            try:
                pl[i] = ml.index(i)
            except ValueError:
                pl[i] = len(ml) + not_ml.index(i)
        x2_tmp = x2_tmp.permute(pl)

        # Element-wise multiplication
        return x2_tmp * x1

    @staticmethod
    def average(x1, x2, ml, el=None):
        """
        Perform an average of the first tensor with the weigths of the second tensor.
        :param x1: the first tensor.
        :param x2: the second tensor.
        :param ml: the maching list describing how the dimensions of the second tensor are
            matched to the dimensions of the first tensor.
        :param el: the elimination list describing which dimension should not be reduced.
        """
        # Create an empty elimination list, if it is None.
        if el is None:
            el = []

        # Perform the element-wise multiplication
        result = Operators.multiplication(x1, x2, ml)

        # Create the reduction list, i.e. rl = ml \ el where "\" = set minus
        rl = ml
        for elem in el:
            rl.remove(elem)

        # Sort the reduction list in decreasing order
        rl.sort(reverse=True)

        # Reduction of the tensor (using a summation) along the dimension of the reduction list
        for i in rl:
            result = result.sum(i)
        return result
