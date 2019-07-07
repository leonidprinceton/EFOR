#MIT License
#
#Copyright (c) 2019, Leonid Pogorelyuk
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import argparse

##
# This is a "post processing" script of a simulated observation of the Wide-Field Infra-Red Survey Telescope (WFIRST) Coronagraph Instrument (CGI).
# The data (.npy files) consists of electric field(s) of the simulated star-light and intensity of a planet, both in the imaging plane of the telescope.
# The script generates images (photon counts) from the electric fields + planet intensity + dark current, then tries to "extract" the signal of the planet. Note that the electric fields change in time due to instrument instabilities.
#
# The estimates of the planet can be computed by (for details, see https://arxiv.org/abs/1907.01801):
# 1. Intensity Principal Component analysis (PCA)  - using two observations, one of the target and one of the reference star
# 2. Electric Field Order Reduction (EFOR)  - using one observation of the target star while randomly actuating the deformable mirrors (dither)
# 3. EFOR with reference data - both target and reference stars observations were simulated with dithering of the deformable mirrors
##

np.random.seed(0)

parser = argparse.ArgumentParser(description="Estimate incoherent intensity from images", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("algorithm", help="algorithm to be used", choices=["PCA","EFOR"])
parser.add_argument("modes", type=int, help="number of modes")
parser.add_argument("-n", help="do not use reference data (EFOR only)", action="store_true")

args = parser.parse_args()

## Loading data simulated FALCO (https://github.com/ajeldorado/falco-matlab). It was originally generated on a 5x68x68 array of wavelengths and pixels, but here only data from 2608 dark-hole pixels at a single wavelength is provided.
incoherent_intensity      = np.load("incoherent_intensity.npy")#planet's PSF in side the dark hole [photons/frame]
Jacobian_target           = np.load("Jacobian_target.npy")#control Jacobian when observing the target star [(photons/frame)**0.5/V]
Jacobian_reference        = np.load("Jacobian_reference.npy")#control Jacobian when observing the reference star [(photons/frame)**0.5/V]
dither_commands_target    = np.load("dither_commands_target.npy")#history of actuations when observing the target star [V]
dither_commands_reference = np.load("dither_commands_reference.npy")#history of actuations when observing the reference star [V]
field_target_dither       = np.load("field_target_dither.npy")#history of electric fields when observing the target star while dithering [(photons/frame)**0.5]
field_reference_dither    = np.load("field_reference_dither.npy")#history of electric fields when observing the reference star while dithering [(photons/frame)**0.5]
field_target_no_dither    = np.load("field_target_no_dither.npy")#history of electric fields when observing the target star without dithering [(photons/frame)**0.5]
field_reference_no_dither = np.load("field_reference_no_dither.npy")#history of electric fields when observing the reference star without dithering [(photons/frame)**0.5]
#pixel_mask               = np.load("pixel_mask.npy")#dark hole 68x68 mask for vizualization purposes
dark_current = 0.25 #[photons/frame]

N = field_target_dither.shape[1]/2#number of pixels in the dark hole; each pixels takes two vector elements for the real and imaginary part of the electric field

## Simulating measured photong counts
ys      = np.random.poisson(field_target_dither[:,::2]**2 + field_target_dither[:,1::2]**2 + incoherent_intensity + dark_current)#photon counts - target start with dithering
y0s     = np.random.poisson(field_target_no_dither[:,::2]**2 + field_target_no_dither[:,1::2]**2 + incoherent_intensity + dark_current)#photon counts - target start without dithering
ys_ref  = np.random.poisson(field_reference_dither[:,::2]**2 + field_reference_dither[:,1::2]**2 + dark_current)#photon counts - reference start with dithering
y0s_ref = np.random.poisson(field_reference_no_dither[:,::2]**2 + field_reference_no_dither[:,1::2]**2 + dark_current)#photon counts - reference start without dithering

def print_error_status(incoherent_intensity_est, info_str):
    intensity_err = np.abs(incoherent_intensity_est-incoherent_intensity)
    mean_everywhere = np.mean(intensity_err) #average error
    mean_half_max = np.mean(intensity_err[incoherent_intensity>0.5*np.max(incoherent_intensity)]) #average error where planet intensity is above half max
    print(info_str + ": mean intensity error - %.03f; mean intensity error in planet's half-max region - %.03f"%(mean_everywhere,mean_half_max))

if args.algorithm == "PCA":
    U,s,V = np.linalg.svd(y0s_ref, full_matrices=False)#modes based on reference data
    P = np.eye(N) - V[:args.modes].T.dot(V[:args.modes])#projection onto the first PCA args.modes modes

    est = P.dot(np.mean(y0s, axis=0))#estimate based on target data
    print_error_status(est*(est>0), "PCA with %d modes"%(args.modes))
else:
    import tensorflow as tf
    from scipy.special import loggamma #can be replaced with np.log and np.math.factorial

    tf.set_random_seed(0)

    ## Defining variables to optimize for the target star
    G0 = np.linalg.svd(np.random.normal(0,1,(args.modes,2*N)), full_matrices=False)[2] #initial guess for the speckles drift field modes (an orthogonal matrix)
    G_t = tf.Variable(G0, dtype=tf.float32) #speckles drift field mode (to be optimized)
    dzs_t = tf.Variable(np.random.normal(0,1,(len(ys),args.modes)), dtype=tf.float32) #history of drift mode coefficients (to be optimized)
    dxs_t = tf.matmul(dzs_t, G_t) #history of open loop field drift
    xs_t = dxs_t + dither_commands_target.dot(Jacobian_target) #history of closed loop (and/or with dithering) field drift

    ## Defining cost function for the target star
    Is_t = tf.square(xs_t[:,::2]) + tf.square(xs_t[:,1::2]) #speckles intensity
    incoherent_intensity_t = tf.nn.relu(tf.reduce_mean(ys - Is_t, axis=0) - dark_current) + dark_current #incoherent intensity estimate which is always >= dark current
    I_t = Is_t + incoherent_intensity_t #total intensity
    log_pp_t = ys*tf.log(I_t) - I_t - loggamma(ys+1) #log probability measuring ys photons given the intensity, I_t
    J_t = -tf.reduce_sum(log_pp_t) #the cost function

    if not args.n: #if reference star data is available
        ## Defining variables to optimize for the reference star
        dzs_ref_t = tf.Variable(np.zeros((len(ys_ref),args.modes)), dtype=tf.float32) #history of drift mode coefficients
        dxs_ref_t = tf.matmul(dzs_ref_t, G_t) #history of open loop field drift
        xs_ref_t = dxs_ref_t + dither_commands_reference.dot(Jacobian_reference) #history of closed loop (and/or with dithering) field drift

        ## Defining cost function for the target star
        Is_ref_t = tf.square(xs_ref_t[:,::2]) + tf.square(xs_ref_t[:,1::2]) #speckles intensity
        incoherent_intensity_ref_t = tf.nn.relu(tf.reduce_mean(ys_ref - Is_ref_t, axis=0) - dark_current) + dark_current #incoherent intensity estimate
        I_ref_t = Is_ref_t + incoherent_intensity_ref_t #total intensity
        log_pp_ref_t = ys_ref*tf.log(I_ref_t) - I_ref_t - loggamma(ys_ref+1) #log probability measuring ys_ref photons given the intensity, I_ref_t

        J_t = J_t - tf.reduce_sum(log_pp_ref_t) #total cost function

    ## Optimizing
    train_op = tf.train.AdamOptimizer(1e-4).minimize(J_t)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2**16+1): #number of iterations
        sess.run(train_op)
        if i%2**10 == 0: #print intermediate results
            EFOR_cost, est = sess.run([J_t, incoherent_intensity_t])
            print_error_status(est-dark_current, "EFOR, %d modes, iteration %d (cost function %.2e)"%(args.modes,i,EFOR_cost))

