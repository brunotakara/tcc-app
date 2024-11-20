# Copy of segmentation_models functional with the addition of the function weighted_f_score to be used in the added WeightedTverskyLoss class

SMOOTH = 1e-5


# ----------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------

def _gather_channels(x, indexes, **kwargs):
    """Slice tensor along channels axis by given indexes"""
    backend = kwargs['backend']
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image, **kwargs):
    backend = kwargs['backend']
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None, **kwargs):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes, **kwargs) for x in xs]
    return xs


def round_if_needed(x, threshold, **kwargs):
    backend = kwargs['backend']
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None, **kwargs):
    backend = kwargs['backend']
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)


# ----------------------------------------------------------------
#   Weighted F Score Function
# ----------------------------------------------------------------

def weighted_f_score(gt, pr, beta=1, eta=1, psi=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None,
            **kwargs):
    r"""
    The formula for the weighted F score in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 \eta FN + \psi FP}

    """

    backend = kwargs['backend']

    gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = round_if_needed(pr, threshold, **kwargs)
    axes = get_reduce_axes(per_image, **kwargs)

    # calculate score
    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    fn = backend.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * eta * fn + psi * fp + smooth)
    score = average(score, per_image, class_weights, **kwargs)

    return score

# ----------------------------------------------------------------
#   Weighted Tversky Loss Function
# ----------------------------------------------------------------


class WeightedTverskyLoss(Loss):
    r"""Creates a criterion to measure weighted Tversy loss, based on weighted dice loss:

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot tp + \beta^2 \cdot \eta \cdot fn + \psi \cdot fp}

    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;

    Args:
        beta: Float or integer coefficient for precision and recall balance.
	eta: fn weight to adjust class imbalance
	psi: fp weight to adjust class imbalance
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
        else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.

    Returns:
        A callable ``weighted_tversy_loss`` instance. Can be used in ``model.compile(...)`` function`
        or combined with other losses.

    Example:

    .. code:: python

        loss = WeightedTverskyLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, beta=1, eta=1, psi=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='weighted_tversky_loss')
        self.beta = beta
        self.eta = eta
        self.psi = psi
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return 1 - F.weighted_f_score(
            gt,
            pr,
            beta=self.beta,
            eta=self.eta,
            psi=self.psi,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules
        )

# Alias

weighted_tversky_loss = WeightedTverskyLoss()

