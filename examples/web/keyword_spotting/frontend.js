// The keyword spotter's audio front end in JavaScript: one second of 16 kHz
// samples to the normalized log-mel spectrogram the model reads, computed as
// keyword_spotting.py's log_mel() and normalize() do with NumPy. The constants
// (mel filters, per-band mean and standard deviation, class names) come from
// kws_frontend.json, which keyword_spotting.py writes next to the model.

export async function loadFrontend(url = "./kws_frontend.json") {
	const response = await fetch(url);
	if (!response.ok)
		throw new Error(`cannot load ${url}: ${response.status}`);
	return makeFrontend(await response.json());
}

export function makeFrontend(fe) {
	const { rate, frame, hop, fft, mels, frames } = fe;
	const bins = fft / 2 + 1;

	// np.hanning(frame): the symmetric Hann window
	const window = new Float64Array(frame);
	for (let n = 0; n < frame; n++)
		window[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (frame - 1));

	// radix-2 FFT tables
	const levels = Math.log2(fft);
	if (!Number.isInteger(levels))
		throw new Error("the FFT size must be a power of two");
	const reverse = new Uint32Array(fft);
	for (let i = 0; i < fft; i++) {
		let r = 0;
		for (let b = 0; b < levels; b++)
			r |= ((i >> b) & 1) << (levels - 1 - b);
		reverse[i] = r;
	}
	const cos = new Float64Array(fft / 2), sin = new Float64Array(fft / 2);
	for (let i = 0; i < fft / 2; i++) {
		cos[i] = Math.cos(2 * Math.PI * i / fft);
		sin[i] = Math.sin(2 * Math.PI * i / fft);
	}
	const re = new Float64Array(fft), im = new Float64Array(fft), power = new Float64Array(bins);

	// the power spectrum |rfft(x * window, fft)|^2 of one frame starting at offset
	function spectrum(samples, offset) {
		re.fill(0);
		im.fill(0);
		for (let n = 0; n < frame; n++)
			re[reverse[n]] = samples[offset + n] * window[n];
		for (let size = 2; size <= fft; size *= 2) {
			const half = size / 2, step = fft / size;
			for (let start = 0; start < fft; start += size)
				for (let k = 0; k < half; k++) {
					const a = start + k, b = a + half;
					const wr = cos[k * step], wi = -sin[k * step];
					const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
					re[b] = re[a] - tr;
					im[b] = im[a] - ti;
					re[a] += tr;
					im[a] += ti;
				}
		}
		for (let k = 0; k < bins; k++)
			power[k] = re[k] * re[k] + im[k] * im[k];
		return power;
	}

	// one second of samples (rate values) to frames x mels normalized log energies,
	// frame by frame: the model's (1, frames, mels) input
	function features(samples) {
		if (samples.length !== rate)
			throw new RangeError(`expected ${rate} samples, got ${samples.length}`);
		const out = new Float32Array(frames * mels);
		for (let t = 0; t < frames; t++) {
			const p = spectrum(samples, t * hop);
			for (let m = 0; m < mels; m++) {
				const { start, weights } = fe.filters[m];
				let energy = 0;
				for (let k = 0; k < weights.length; k++)
					energy += p[start + k] * weights[k];
				out[t * mels + m] = (Math.log(energy + 1e-6) - fe.mean[m]) / fe.std[m];
			}
		}
		return out;
	}

	return { ...fe, features };
}
