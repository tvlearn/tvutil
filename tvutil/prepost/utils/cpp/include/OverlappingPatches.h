/* Copyright (C) 2024 Machine Learning Lab of the University of Oldenburg. */
/* Licensed under the Academic Free License version 3.0                    */

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <Eigen/Core>
#include <iostream>

#include "Numpy.h"

class OverlappingPatches {
   public:
    long no_dim;
    long no_pixels_to_synthesize;
    long max_pixels_to_restore;
    long patch_shift;

    Matrix<long> ind_to_synthesize;
    Vector<long> image_shapes;
    Vector<long> patch_shapes;

    Vector<long> no_patches_per_axis;
    Vector<long> no_patches_per_axis_shift_1;
    Vector<long> img_skips;
    Vector<long> img_skips_shift_1;
    Vector<long> patch_skips;

    OverlappingPatches(long, long, cRef<Matrix<long>>, cRef<Vector<long>>, cRef<Vector<long>>,
                       cRef<Vector<long>>, cRef<Vector<long>>);

    void set_merge_method(std::string);

    void createRange(std::vector<long> &, std::vector<bool> &, long, long, long, long);
    void createRange(std::vector<long> &, long, long);

    template <class T>
    void cartesianProduct(const std::vector<std::vector<T>> &lists, Ref<Matrix<T>> output, long rows,
                          long cols);

    long back_transformation(long p, Ref<Vector<long>> all_inds_relevant_patches,
                             Ref<Vector<long>> all_inds_relevant_values_in_patch, Ref<Matrix<long>> _ns,
                             Ref<Matrix<long>> _ds, Ref<Matrix<bool>> _to_keep, Ref<Vector<bool>> nsinds,
                             std::vector<std::vector<long>> &loc_rel_patches,
                             std::vector<std::vector<long>> &loc_rel_patches_,
                             std::vector<std::vector<bool>> &loc_to_keep);

    precision_t (*merger)(Ref<Vector<precision_t>>);

    void merge(cRef<Matrix<>> patches, Ref<Vector<>> new_image, bool no_empty_patches, bool image_complete, std::string merge_method);

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
    static void bind(py::module_ &m);
#endif
};

//--------------------------------------------------------------------------------------------------------------------//

precision_t median(Ref<Vector<precision_t>> v) {
    long n = (v.size() / 2) + (v.size() % 2);
    std::nth_element(v.begin(), v.begin() + n, v.end());
    if (v.size() % 2 == 0) {
        return 0.5 * (v[n - 1] + v.tail(n).minCoeff());
    } else {
        return v[n - 1];
    }
}

precision_t mean(Ref<Vector<precision_t>> v) { return v.mean(); }

precision_t max(Ref<Vector<precision_t>> v) { return v.maxCoeff(); }

precision_t min(Ref<Vector<precision_t>> v) { return v.minCoeff(); }

precision_t variance(Ref<Vector<precision_t>> v) {  // ToDo: Implement ddof (compare np.var)
    precision_t mean = v.mean();
    return (v.array() - mean).square().sum() / (v.size());
}

OverlappingPatches::OverlappingPatches(long _no_pixels_to_synthesize, long _patch_shift,
                                       cRef<Matrix<long>> _ind_to_synthesize, cRef<Vector<long>> _image_shapes,
                                       cRef<Vector<long>> _patch_shapes,
                                       cRef<Vector<long>> _no_patches_per_axis,
                                       cRef<Vector<long>> _no_patches_per_axis_shift_1) :
    no_pixels_to_synthesize(_no_pixels_to_synthesize),
    patch_shift(_patch_shift),
    ind_to_synthesize(_ind_to_synthesize),
    image_shapes(_image_shapes),
    patch_shapes(_patch_shapes),
    no_patches_per_axis(_no_patches_per_axis),
    no_patches_per_axis_shift_1(_no_patches_per_axis_shift_1),
    img_skips(_image_shapes.size()),
    img_skips_shift_1(_image_shapes.size()),
    patch_skips(_image_shapes.size()) {
    no_dim = image_shapes.size();
    max_pixels_to_restore = patch_shapes.prod();
    for (long i = 1; i < no_dim; ++i) {
        img_skips[i - 1] = no_patches_per_axis.tail(no_dim - i).prod();
        img_skips_shift_1[i - 1] = image_shapes.tail(no_dim - i).prod();
        patch_skips[i - 1] = patch_shapes.tail(no_dim - i).prod();
    }
    img_skips[no_dim - 1] = 1;
    img_skips_shift_1[no_dim - 1] = 1;
    patch_skips[no_dim - 1] = 1;
}

void OverlappingPatches::set_merge_method(std::string merge_method) {
    if (merge_method == "median") {
        merger = median;
        return;
    }
    if (merge_method == "mean") {
        merger = mean;
        return;
    }
    if (merge_method == "max") {
        merger = max;
        return;
    }
    if (merge_method == "min") {
        merger = min;
        return;
    }
    if (merge_method == "variance") {
        merger = variance;
        return;
    }
    std::string err = "merge method '" + merge_method + "' is not implemented!";
    throw std::runtime_error(err);
}

// Helper function to generate a range with specified boundaries
void OverlappingPatches::createRange(std::vector<long> &range, std::vector<bool> &check, long start, long end,
                                     long last_patch, long patch_shift) {
    range.clear();
    check.clear();
    for (long i = start; i < end; ++i) {
        if ((i == (last_patch - 1)) | (i % patch_shift == 0)) {
            check.push_back(true);
        } else {
            check.push_back(false);
        }
        range.push_back(i);
    }
    return;
}

void OverlappingPatches::createRange(std::vector<long> &range, long start, long end) {
    range.clear();
    for (long i = start; i < end; ++i) {
        range.push_back(i);
    }
    return;
}

// Helper function to generate Cartesian product from a vector of ranges
template <class T>
void OverlappingPatches::cartesianProduct(const std::vector<std::vector<T>> &lists, Ref<Matrix<T>> output,
                                          long end, long no_dim) {
    long repeat = 1;
    for (long j = no_dim - 1; j >= 0; --j) {
        const auto &list = lists[j];
        for (long i = 0; i < end; ++i) {
            output(i, j) = list[(i / repeat) % list.size()];
        }
        repeat *= list.size();
    }
}

long OverlappingPatches::back_transformation(
    long p, Ref<Vector<long>> all_inds_relevant_patches, Ref<Vector<long>> all_inds_relevant_values_in_patch,
    Ref<Matrix<long>> n_inds_per_axis, Ref<Matrix<long>> d_inds_per_axis,
    Ref<Matrix<bool>> n_inds_per_axis_to_keep, Ref<Vector<bool>> n_inds_to_keep,
    std::vector<std::vector<long>> &loc_rel_patches, std::vector<std::vector<long>> &loc_rel_values_in_patch,
    std::vector<std::vector<bool>> &loc_to_keep) {
    long loc_miss_value;
    long shp;
    long no_shift;
    long end = 1;

    // Compute indices of relevant patches and relevant values per patch for each axis
    for (long i = 0; i < no_dim; ++i) {
        loc_miss_value = ind_to_synthesize(i, p);
        shp = patch_shapes[i];
        no_shift = no_patches_per_axis_shift_1[i];

        if (patch_shift > 1) {
            createRange(loc_rel_patches[i], loc_to_keep[i], std::max(loc_miss_value - shp + 2, (long)1) - 1,
                        std::min(loc_miss_value + 1, no_shift), no_shift, patch_shift);
        } else {
            createRange(loc_rel_patches[i], std::max(loc_miss_value - shp + 2, (long)1) - 1,
                        std::min(loc_miss_value + 1, no_shift));
        }

        createRange(loc_rel_values_in_patch[i], loc_miss_value - std::min(loc_miss_value + 1, no_shift) + 1,
                    loc_miss_value - (std::max(loc_miss_value - shp + 2, (long)1) - 1) + 1);
        end *= (long)loc_rel_patches[i].size();
    }

    // Combine all relevant patches and relevant values per patch for each axis
    cartesianProduct(loc_rel_patches, n_inds_per_axis, end, no_dim);
    cartesianProduct(loc_rel_values_in_patch, d_inds_per_axis, end, no_dim);

    // Compute indices for relevant values per patch flattened for patch_shift = 1
    all_inds_relevant_values_in_patch.head(end) =
        (d_inds_per_axis.block(0, 0, end, no_dim).array().rowwise() * patch_skips.array())
            .rowwise()
            .sum()
            .reverse();

    if (patch_shift > 1) {
        // Compute which patches to keep for patch_shift > 1
        cartesianProduct(loc_to_keep, n_inds_per_axis_to_keep, end, no_dim);
        n_inds_to_keep = n_inds_per_axis_to_keep.array().rowwise().prod().cast<bool>();
        long new_end = n_inds_to_keep.head(end).count();

        // Compute indices for relevant values per patch flattened for patch_shift > 1
        long idx = 0;
        for (long i = 0; i < end; ++i) {
            if (n_inds_to_keep(i)) {
                n_inds_per_axis.row(idx) = n_inds_per_axis.row(i);
                all_inds_relevant_values_in_patch(idx) = all_inds_relevant_values_in_patch(i);
                ++idx;
            }
        }
        // Compute indices for relevant patches flattened for patch_shift > 1
        all_inds_relevant_patches.head(new_end) =
            (((n_inds_per_axis.array().block(0, 0, new_end, no_dim) + patch_shift - 1) / patch_shift)
                 .rowwise() *
             img_skips.array())
                .rowwise()
                .sum();
        end = new_end;
    } else {
        // Compute indices for relevant patches flattened for patch_shift = 1
        all_inds_relevant_patches.head(end) =
        (n_inds_per_axis.block(0, 0, end, no_dim).array().rowwise() * img_skips.array()).rowwise().sum();
    }
    return end;
}

void OverlappingPatches::merge(cRef<Matrix<>> patches, Ref<Vector<>> new_image,  bool no_empty_patches, bool image_complete, std::string merge_method
                               // bool verbose,
) {
    set_merge_method(merge_method);

    bool all_pixels_reconstructed = true;
    long no_dim = patch_shapes.size();
#pragma omp parallel
    {
        bool all_pixels_reconstructed_thread = true;
        Vector<> restored(max_pixels_to_restore);
        restored.fill(0.0);
        precision_t estimate;

        Matrix<long> n_inds_per_axis(max_pixels_to_restore, no_dim);
        Matrix<long> d_inds_per_axis(max_pixels_to_restore, no_dim);
        Matrix<bool> n_inds_per_axis_to_keep(max_pixels_to_restore, no_dim);

        Vector<bool> n_inds_to_keep(max_pixels_to_restore);

        std::vector<std::vector<long>> loc_rel_patches(no_dim);
        std::vector<std::vector<long>> loc_rel_values_in_patch(no_dim);
        std::vector<std::vector<bool>> loc_to_keep(no_dim);

        Vector<long> all_inds_relevant_patches(max_pixels_to_restore);          // Relevant patches per pixel
        Vector<long> all_inds_relevant_values_in_patch(max_pixels_to_restore);  // Relevant values in patch
        long end;
        long new_end;
        long idx;

#pragma omp for
        for (long p = 0; p < no_pixels_to_synthesize; ++p) {
            end = back_transformation(p, all_inds_relevant_patches, all_inds_relevant_values_in_patch,
                                      n_inds_per_axis, d_inds_per_axis, n_inds_per_axis_to_keep,
                                      n_inds_to_keep, loc_rel_patches, loc_rel_values_in_patch, loc_to_keep);

            if (end == 0) {
                all_pixels_reconstructed_thread = false;
                continue;
            }

            if (no_empty_patches) {
                for (long i = 0; i < end; ++i) {
                    restored(i) = patches(all_inds_relevant_patches(i), all_inds_relevant_values_in_patch(i));
                }
            } else {
                new_end = 0;
                for (long i = 0; i < end; ++i) {
                    if (patches.row(all_inds_relevant_patches(i)).array().isFinite().any()) {
                        restored(new_end) = patches(all_inds_relevant_patches(i), all_inds_relevant_values_in_patch(i));
                        new_end++;
                    }
                }
                end = new_end;
                if (end == 0) {
                    all_pixels_reconstructed_thread = false;
                    continue;
                }
            }

            estimate = merger(restored.head(end));
            
            if (!image_complete){
                idx = 0;
                for (long i = 0; i < ind_to_synthesize.rows(); ++i) {
                    idx += ind_to_synthesize(i, p) * img_skips_shift_1(i);
                }
            } else {
                idx = p;
            }
            new_image(idx) = estimate;
        }
#pragma omp atomic
        all_pixels_reconstructed &= all_pixels_reconstructed_thread;
    }
    if (!all_pixels_reconstructed) {
        std::cout << "\nWARNING: Not all pixels in the image have been reconstructed!\n";
    }
}

#ifdef CPPLIB_ENABLE_PYTHON_INTERFACE
void OverlappingPatches::bind(pybind11::module_ &m) {
    pybind11::class_<OverlappingPatches> OverlappingPatches_class_(m, "OverlappingPatches",
                                                                   pybind11::module_local());

    OverlappingPatches_class_.def(py::init<long, long, cRef<Matrix<long>>, cRef<Vector<long>>,
                                           cRef<Vector<long>>, cRef<Vector<long>>, cRef<Vector<long>>>(),
                                  "no_pixels_to_synthesize"_a, "patch_shift"_a, "ind_to_synthesize"_a,
                                  "image_shapes"_a, "patch_shapes"_a, "no_patches_per_axis"_a,
                                  "no_patches_per_axis_shift_1"_a);

    OverlappingPatches_class_.def("merge", &OverlappingPatches::merge, "patches"_a.noconvert(),
                                  "new_image"_a.noconvert(), "no_empty_patches"_a, "image_complete"_a, "merge_method"_a = "mean");
}
#endif
