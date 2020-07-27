library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;

package fixed_pkg is new IEEE.fixed_generic_pkg
  generic map (
    fixed_round_style    => IEEE.fixed_float_types.fixed_round,
    fixed_overflow_style => IEEE.fixed_float_types.fixed_saturate,
    fixed_guard_bits     => 3,
    no_warning           => false
    );

use work.fixed_pkg.all;
use work.pkg_param.all;

package pkg_types is

  subtype t_data is sfixed(DATA_INT_BW - 1 downto -DATA_FRAC_BW);
  type t_feature_map_row is array(0 to IN_FM_WIDTH - 1) of t_data;
  type t_feature_map is array(0 to IN_FM_HEIGHT - 1) of t_feature_map_row;
  type t_data_tensor is array(0 to NUM_CHANNELS - 1) of t_feature_map;

  type t_conv2d_feature_map_row is array(0 to IN_FM_WIDTH + 1) of t_data;
  type t_conv2d_feature_map is array(0 to IN_FM_HEIGHT + 1) of t_conv2d_feature_map_row;
  type t_conv2d_data_tensor is array(0 to NUM_CHANNELS - 1) of t_conv2d_feature_map;

  subtype t_weight is sfixed(WEIGHT_INT_BW - 1 downto -WEIGHT_FRAC_BW);
  type t_kernel_row is array(0 to KERNEL_SIZE - 1) of t_weight;
  type t_kernel_matrix is array(0 to KERNEL_SIZE - 1) of t_kernel_row;
  type t_kernel_tensor is array(0 to NUM_CHANNELS - 1) of t_kernel_matrix;
  type t_kernels_layer is array(0 to NUM_CHANNELS - 1) of t_kernel_tensor;

  type t_history is array(0 to HISTORY_DEPTH - 1) of t_feature_map_row;

  type t_fc_input is array(0 to NUM_CHANNELS - 1) of t_data;
  type t_fc_output is array(0 to NUM_CLASSES - 1) of t_data;

  subtype t_bias is sfixed(BIAS_INT_BW - 1 downto -BIAS_FRAC_BW);
  type t_biases_array is array(0 to NUM_CHANNELS - 1) of t_bias;
  type t_biases_matrix is array(0 to NUM_ITERATIONS - 1) of t_biases_array;

  subtype t_gamma is sfixed(GAMMA_INT_BW - 1 downto -GAMMA_FRAC_BW);
  type t_gammas_row is array(0 to IN_FM_WIDTH - 1) of t_gamma;
  type t_gammas_matrix is array(0 to IN_FM_HEIGHT - 1) of t_gammas_row;
  type t_gammas_tensor is array(0 to NUM_CHANNELS - 1) of t_gammas_matrix;
  type t_gammas is array(0 to NUM_ITERATIONS - 1) of t_gammas_tensor;


end pkg_types;
