#From https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/processing/features.html#Filterbank

import math
import torch
import logging

class Filterbank(torch.nn.Module):
    """computes filter bank (FBANK) features given spectral magnitudes.

    Arguments
    ---------
    n_mels : float
        Number of Mel filters used to average the spectrogram.
    log_mel : bool
        If True, it computes the log of the FBANKs.
    filter_shape : str
        Shape of the filters ('triangular', 'rectangular', 'gaussian').
    f_min : int
        Lowest frequency for the Mel filters.
    f_max : int
        Highest frequency for the Mel filters.
    n_fft : int
        Number of fft points of the STFT. It defines the frequency resolution
        (n_fft should be<= than win_len).
    sample_rate : int
        Sample rate of the input audio signal (e.g, 16000)
    power_spectrogram : float
        Exponent used for spectrogram computation.
    amin : float
        Minimum amplitude (used for numerical stability).
    ref_value : float
        Reference value used for the dB scale.
    top_db : float
        Minimum negative cut-off in decibels.
    freeze : bool
        If False, it the central frequency and the band of each filter are
        added into nn.parameters. If True, the standard frozen features
        are computed.
    param_change_factor: bool
        If freeze=False, this parameter affects the speed at which the filter
        parameters (i.e., central_freqs and bands) can be changed.  When high
        (e.g., param_change_factor=1) the filters change a lot during training.
        When low (e.g. param_change_factor=0.1) the filter parameters are more
        stable during training
    param_rand_factor: float
        This parameter can be used to randomly change the filter parameters
        (i.e, central frequencies and bands) during training.  It is thus a
        sort of regularization. param_rand_factor=0 does not affect, while
        param_rand_factor=0.15 allows random variations within +-15% of the
        standard values of the filter parameters (e.g., if the central freq
        is 100 Hz, we can randomly change it from 85 Hz to 115 Hz).

    Example
    -------
    >>> import torch
    >>> compute_fbanks = Filterbank()
    >>> inputs = torch.randn([10, 101, 201])
    >>> features = compute_fbanks(inputs)
    >>> features.shape
    torch.Size([10, 101, 40])
    """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape="triangular",
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.0,
        top_db=80.0,
        param_change_factor=1.0,
        param_rand_factor=0.0,
        freeze=True,
        **kwargs
    ):
        super().__init__()
        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.n_stft = self.n_fft // 2 + 1
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        self.device_inp = torch.device("cpu")
        self.param_change_factor = param_change_factor
        self.param_rand_factor = param_rand_factor

        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        # Make sure f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logger.error(err_msg, exc_info=True)

        # Filter definition
        mel = torch.linspace(
            self._to_mel(self.f_min), self._to_mel(self.f_max), self.n_mels + 2
        )
        hz = self._to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]
        self.band = band[:-1]
        self.f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        if not self.freeze:
            self.f_central = torch.nn.Parameter(
                self.f_central / (self.sample_rate * self.param_change_factor)
            )
            self.band = torch.nn.Parameter(
                self.band / (self.sample_rate * self.param_change_factor)
            )

        # Frequency axis
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # Replicating for all the filters
        self.all_freqs_mat = all_freqs.repeat(self.f_central.shape[0], 1)

    def forward(self, spectrogram):
        """Returns the FBANks.

        Arguments
        ---------
        x : tensor
            A batch of spectrogram tensors.
        """
        # Computing central frequency and bandwidth of each filter
        f_central_mat = self.f_central.repeat(
            self.all_freqs_mat.shape[1], 1
        ).transpose(0, 1)
        band_mat = self.band.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )

        # Uncomment to print filter parameters
        # print(self.f_central*self.sample_rate * self.param_change_factor)
        # print(self.band*self.sample_rate* self.param_change_factor)

        # Creation of the multiplication matrix. It is used to create
        # the filters that average the computed spectrogram.
        if not self.freeze:
            f_central_mat = f_central_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )
            band_mat = band_mat * (
                self.sample_rate
                * self.param_change_factor
                * self.param_change_factor
            )

        # Regularization with random changes of filter central frequency and band
        elif self.param_rand_factor != 0 and self.training:
            rand_change = (
                1.0
                + torch.rand(2) * 2 * self.param_rand_factor
                - self.param_rand_factor
            )
            f_central_mat = f_central_mat * rand_change[0]
            band_mat = band_mat * rand_change[1]

        fbank_matrix = self._create_fbank_matrix(f_central_mat, band_mat).to(
            spectrogram.device
        )

        sp_shape = spectrogram.shape

        # Managing multi-channels case (batch, time, channels)
        if len(sp_shape) == 4:
            spectrogram = spectrogram.permute(0, 3, 1, 2)
            spectrogram = spectrogram.reshape(
                sp_shape[0] * sp_shape[3], sp_shape[1], sp_shape[2]
            )

        # FBANK computation
        fbanks = torch.matmul(spectrogram, fbank_matrix)
        if self.log_mel:
            fbanks = self._amplitude_to_DB(fbanks)

        # Reshaping in the case of multi-channel inputs
        if len(sp_shape) == 4:
            fb_shape = fbanks.shape
            fbanks = fbanks.reshape(
                sp_shape[0], sp_shape[3], fb_shape[1], fb_shape[2]
            )
            fbanks = fbanks.permute(0, 2, 3, 1)

        return fbanks


    @staticmethod
    def _to_mel(hz):
        """Returns mel-frequency value corresponding to the input
        frequency value in Hz.

        Arguments
        ---------
        x : float
            The frequency point in Hz.
        """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        """Returns hz-frequency value corresponding to the input
        mel-frequency value.

        Arguments
        ---------
        x : float
            The frequency point in the mel-scale.
        """
        return 700 * (10 ** (mel / 2595) - 1)

    def _triangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using triangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # Computing the slops of the filters
        slope = (all_freqs - f_central) / band
        left_side = slope + 1.0
        right_side = -slope + 1.0

        # Adding zeros for negative values
        zero = torch.zeros(1, device=self.device_inp)
        fbank_matrix = torch.max(
            zero, torch.min(left_side, right_side)
        ).transpose(0, 1)

        return fbank_matrix

    def _rectangular_filters(self, all_freqs, f_central, band):
        """Returns fbank matrix using rectangular filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        """

        # cut-off frequencies of the filters
        low_hz = f_central - band
        high_hz = f_central + band

        # Left/right parts of the filter
        left_side = right_size = all_freqs.ge(low_hz)
        right_size = all_freqs.le(high_hz)

        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def _gaussian_filters(
        self, all_freqs, f_central, band, smooth_factor=torch.tensor(2)
    ):
        """Returns fbank matrix using gaussian filters.

        Arguments
        ---------
        all_freqs : Tensor
            Tensor gathering all the frequency points.
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        fbank_matrix = torch.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def _create_fbank_matrix(self, f_central_mat, band_mat):
        """Returns fbank matrix to use for averaging the spectrum with
           the set of filter-banks.

        Arguments
        ---------
        f_central : Tensor
            Tensor gathering central frequencies of each filter.
        band : Tensor
            Tensor gathering the bands of each filter.
        smooth_factor: Tensor
            Smoothing factor of the gaussian filter. It can be used to employ
            sharper or flatter filters.
        """
        if self.filter_shape == "triangular":
            fbank_matrix = self._triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        elif self.filter_shape == "rectangular":
            fbank_matrix = self._rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        else:
            fbank_matrix = self._gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def _amplitude_to_DB(self, x):
        """Converts  linear-FBANKs to log-FBANKs.

        Arguments
        ---------
        x : Tensor
            A batch of linear FBANK tensors.

        """

        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max. It is the max over time and frequency,
        # Hence, of a whole sequence (sequence-dependent)
        new_x_db_max = x_db.amax(dim=(-2, -1)) - self.top_db

        # Clipping to dB max. The view is necessary as only a scalar is obtained
        # per sequence.
        x_db = torch.max(x_db, new_x_db_max.view(x_db.shape[0], 1, 1))

        return x_db

class STFT(torch.nn.Module):
    """computes the Short-Term Fourier Transform (STFT).

    This class computes the Short-Term Fourier Transform of an audio signal.
    It supports multi-channel audio inputs (batch, time, channels).

    Arguments
    ---------
    sample_rate : int
        Sample rate of the input audio signal (e.g 16000).
    win_length : float
        Length (in ms) of the sliding window used to compute the STFT.
    hop_length : float
        Length (in ms) of the hope of the sliding window used to compute
        the STFT.
    n_fft : int
        Number of fft point of the STFT. It defines the frequency resolution
        (n_fft should be <= than win_len).
    window_fn : function
        A function that takes an integer (number of samples) and outputs a
        tensor to be multiplied with each window before fft.
    normalized_stft : bool
        If True, the function returns the  normalized STFT results,
        i.e., multiplied by win_length^-0.5 (default is False).
    center : bool
        If True (default), the input will be padded on both sides so that the
        t-th frame is centered at time t×hop_length. Otherwise, the t-th frame
        begins at time t×hop_length.
    pad_mode : str
        It can be 'constant','reflect','replicate', 'circular', 'reflect'
        (default). 'constant' pads the input tensor boundaries with a
        constant value. 'reflect' pads the input tensor using the reflection
        of the input boundary. 'replicate' pads the input tensor using
        replication of the input boundary. 'circular' pads using  circular
        replication.
    onesided : True
        If True (default) only returns nfft/2 values. Note that the other
        samples are redundant due to the Fourier transform conjugate symmetry.

    Example
    -------
    >>> import torch
    >>> compute_STFT = STFT(
    ...     sample_rate=16000, win_length=25, hop_length=10, n_fft=400
    ... )
    >>> inputs = torch.randn([10, 16000])
    >>> features = compute_STFT(inputs)
    >>> features.shape
    torch.Size([10, 101, 201, 2])
    """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        self.window = window_fn(self.win_length)

    def forward(self, x):
        """Returns the STFT generated from the input waveforms.

        Arguments
        ---------
        x : tensor
            A batch of audio signals to transform.
        """

        # Managing multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])

        #support for older version is not necessary. Based on requirements.txt we can assume torch >=1.7
        # if version.parse(torch.__version__) <= version.parse("1.6.0"):
        #     stft = torch.stft(
        #         x,
        #         self.n_fft,
        #         self.hop_length,
        #         self.win_length,
        #         self.window.to(x.device),
        #         self.center,
        #         self.pad_mode,
        #         self.normalized_stft,
        #         self.onesided,
        #     )
        # else:
        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(x.device),
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
            return_complex=False,
            )

        # Retrieving the original dimensionality (batch,time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            # (batch, time, channels)
            stft = stft.transpose(2, 1)

        return stft

class Deltas(torch.nn.Module):
    """Computes delta coefficients (time derivatives).

    Arguments
    ---------
    win_length : int
        Length of the window used to compute the time derivatives.

    Example
    -------
    >>> inputs = torch.randn([10, 101, 20])
    >>> compute_deltas = Deltas(input_size=inputs.size(-1))
    >>> features = compute_deltas(inputs)
    >>> features.shape
    torch.Size([10, 101, 20])
    """

    def __init__(
        self, input_size, window_length=5,
    ):
        super().__init__()
        self.n = (window_length - 1) // 2
        self.denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3

        self.register_buffer(
            "kernel",
            torch.arange(-self.n, self.n + 1, dtype=torch.float32,).repeat(
                input_size, 1, 1
            ),
        )

    def forward(self, x):
        """Returns the delta coefficients.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """
        # Managing multi-channel deltas reshape tensor (batch*channel,time)
        x = x.transpose(1, 2).transpose(2, -1)
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Padding for time borders
        x = torch.nn.functional.pad(x, (self.n, self.n), mode="replicate")

        # Derivative estimation (with a fixed convolutional kernel)
        delta_coeff = (
            torch.nn.functional.conv1d(
                x, self.kernel.to(x.device), groups=x.shape[1]
            )
            / self.denom
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            delta_coeff = delta_coeff.reshape(
                or_shape[0], or_shape[1], or_shape[2], or_shape[3],
            )
        delta_coeff = delta_coeff.transpose(1, -1).transpose(2, -1)

        return delta_coeff

class ContextWindow(torch.nn.Module):
    """Computes the context window.

    This class applies a context window by gathering multiple time steps
    in a single feature vector. The operation is performed with a
    convolutional layer based on a fixed kernel designed for that.

    Arguments
    ---------
    left_frames : int
         Number of left frames (i.e, past frames) to collect.
    right_frames : int
        Number of right frames (i.e, future frames) to collect.

    Example
    -------
    >>> import torch
    >>> compute_cw = ContextWindow(left_frames=5, right_frames=5)
    >>> inputs = torch.randn([10, 101, 20])
    >>> features = compute_cw(inputs)
    >>> features.shape
    torch.Size([10, 101, 220])
    """

    def __init__(
        self, left_frames=0, right_frames=0,
    ):
        super().__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.context_len = self.left_frames + self.right_frames + 1
        self.kernel_len = 2 * max(self.left_frames, self.right_frames) + 1

        # Kernel definition
        self.kernel = torch.eye(self.context_len, self.kernel_len)

        if self.right_frames > self.left_frames:
            lag = self.right_frames - self.left_frames
            self.kernel = torch.roll(self.kernel, lag, 1)

        self.first_call = True

    def forward(self, x):
        """Returns the tensor with the surrounding context.

        Arguments
        ---------
        x : tensor
            A batch of tensors.
        """

        x = x.transpose(1, 2)

        if self.first_call is True:
            self.first_call = False
            self.kernel = (
                self.kernel.repeat(x.shape[1], 1, 1)
                .view(x.shape[1] * self.context_len, self.kernel_len,)
                .unsqueeze(1)
            )

        # Managing multi-channel case
        or_shape = x.shape
        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Compute context (using the estimated convolutional kernel)
        cw_x = torch.nn.functional.conv1d(
            x,
            self.kernel.to(x.device),
            groups=x.shape[1],
            padding=max(self.left_frames, self.right_frames),
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            cw_x = cw_x.reshape(
                or_shape[0], cw_x.shape[1], or_shape[2], cw_x.shape[-1]
            )

        cw_x = cw_x.transpose(1, 2)

        return cw_x

def spectral_magnitude(stft, power=1, log=False, eps=1e-14):
    """Returns the magnitude of a complex spectrogram.

    Arguments
    ---------
    stft : torch.Tensor
        A tensor, output from the stft function.
    power : int
        What power to use in computing the magnitude.
        Use power=1 for the power spectrogram.
        Use power=0.5 for the magnitude spectrogram.
    log : bool
        Whether to apply log to the spectral features.

    Example
    -------
    >>> a = torch.Tensor([[3, 4]])
    >>> spectral_magnitude(a, power=0.5)
    tensor([5.])
    """
    spectr = stft.pow(2).sum(-1)

    # Add eps avoids NaN when spectr is zero
    if power < 1:
        spectr = spectr + eps
    spectr = spectr.pow(power)

    if log:
        return torch.log(spectr + eps)
    return spectr