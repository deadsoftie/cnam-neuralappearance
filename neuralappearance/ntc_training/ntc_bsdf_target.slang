// SPDX-License-Identifier: Apache-2.0

// BSDF target of the E2E loss: the dense data generator's prefiltered target,
// evaluated once at mip 0, where the prefilter width is zero and every
// prefilter sample lands on the same point.
__exported import datagen.falcor_bsdf_data_generator;

void ntc_bsdf_target(
    DataGenerator::Config config,
    int texture_resolution,
    inout UniformSampleGenerator sg,
    inout TrainingSample<3, 1> sample)
{
    if (sample.mip_level == 0)
        DataGenerator::Bsdf::Falcor::fill_target_bsdf<1>(config, texture_resolution, sg, sample);
    else
        DataGenerator::Bsdf::Falcor::fill_target_bsdf_lod<1>(config, texture_resolution, sg, sample);
}
