# GAN-based PDF Learning

**Student:** Naman Singh  
**Roll:** 102317144  
**Email:** nsingh1_be23@thapar.edu

## What This Assignment is About

Had to learn an unknown probability density function using only data samples. No analytical form given - just transform NO2 data using a specific function and train a GAN to figure out the distribution.

## My Transformation Parameters

Based on roll number 102317144:

```
a_r = 0.5 × (102317144 mod 7) = 0.5 × 6 = 3.0
b_r = 0.3 × (102317144 mod 5 + 1) = 0.3 × 5 = 1.5
```

So my transformation function is:
```
z = x + 3.0 × sin(1.5 × x)
```

The sine component creates oscillations that make the transformed distribution weird. You can't use simple Gaussian or exponential models for this.

## Dataset I Used

Used India Air Quality dataset - basically NO2 concentration readings.

- Total samples: 419,509
- Actually used: 50,000 (random subset)
- Range: 0 to 876 μg/m³
- Mean: around 25.81

I subsampled because 50k samples is plenty to learn the distribution. Don't really need millions of samples for this.

## GAN Architecture

Made two networks:

**Generator:**
- Takes 100-dim noise from N(0,1)
- Has 256 → 128 hidden units
- Uses LeakyReLU activation
- Batch norm after each layer
- Outputs 1 value (the z sample)

Basically maps random noise to realistic z values.

**Discriminator:**
- Takes 1 value as input
- Has 128 → 64 hidden units
- LeakyReLU activation
- Dropout for regularization
- Outputs probability that input is real

Tries to tell real from fake samples.

## Training

- Ran for 1000 epochs
- Batch size: 256
- Used Adam optimizer (lr=0.0002)
- BCE loss
- Normalized the data first

Took about 8-10 minutes on Colab GPU.

## What I Observed

### Training Behavior

Training was stable. Generator loss stayed around 0.6-0.9 and discriminator  went up to 1.38 (but that's okay).

Both losses fluctuate a bit which is good - means they're competing properly. If they go flat too early that's mode collapse.

Adding batch normalization helped a lot. Without it the generator was pretty unstable early on.

### Mode Coverage

Looking at the plots, the GAN learned most of the distribution modes well.

Real distribution has peaks and valleys from the sine transformation. Generated distribution matches these peaks pretty well. No obvious mode collapse - generator produces varied outputs.

Some differences in the tails but GANs usually struggle with rare extreme values anyway.

### How Good the Results Are

Histogram shows good overlap between real and generated. Main shape matches.

Q-Q plot is mostly linear so quantiles match well. Some deviation at extremes but overall accurate.

KDE curves follow similar patterns. Generated is slightly smoother which makes sense.

Stats:
- Real mean: 28.27, Generated: 28.15 (close)
- Real std: 20.81, Generated: 20.34 (similar)

### What Worked

Batch normalization was key for stability.

LeakyReLU instead of ReLU helped too - avoids dead neurons.

Learning rate 0.0002 worked fine.

### What Could Be Better

Extreme tails don't match perfectly. Known GAN issue - they focus on the bulk.

Could probably get better with more epochs (like 2000) but 1000 was enough to see convergence.

Wasserstein GAN might give better stability but assignment didn't specify which type to use.

## PDF Estimation

Generated 50,000 samples from trained generator.

Used KDE with Gaussian kernel to get the PDF. Basically creates a smooth curve from discrete samples.

The PDF shows it's not a standard distribution - has multiple modes and asymmetry from sine transformation.

## Files

- `gan_messy.ipynb` - Main code
- `README.md` - This file
- `results.png` - All plots

## How to Run

1. Upload to Colab
2. Enable GPU (Runtime → Change runtime type → GPU)
3. Upload data.csv
4. Run cells

Takes 8-10 minutes with GPU.

## Main Takeaway

This showed how GANs can learn complex distributions without knowing the analytical form. Generator implicitly models the PDF just from samples.

For transformed NO2 data with sine component, no standard model would work. But GAN's neural network can approximate any function so it learned the weird shape.

The adversarial training forces generator to match real distribution. If it doesn't, discriminator catches it. This makes generator keep improving.

Pretty cool that you can estimate PDF this way without assumptions about distribution type.

---

**Transformation:** z = x + 3.0×sin(1.5×x) where x is NO2 concentration

**GAN Details:**
- Generator: Noise(100) → Dense(256) → BN → LeakyReLU → Dense(128) → BN → LeakyReLU → Dense(1)
- Discriminator: Input(1) → Dense(128) → LeakyReLU → Dropout → Dense(64) → LeakyReLU → Dropout → Dense(1) → Sigmoid
