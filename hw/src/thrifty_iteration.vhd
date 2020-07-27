library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity thrifty_iteration is
  port(
    clk : in std_logic;

    fm_height : in natural range 1 to IN_FM_HEIGHT;
    fm_width  : in natural range 1 to IN_FM_WIDTH;

    iter_index : in natural range 0 to NUM_ITERATIONS - 1;
    ds_enable : in std_logic;

    input_tensor  : in  t_data_tensor;
    output_tensor : out t_data_tensor

    );
end entity thrifty_iteration;

architecture rtl of thrifty_iteration is

  signal conv2d_output : t_data_tensor;
  signal relu_output : t_data_tensor;
  signal batchnorm_output : t_data_tensor;
  signal res_add_output : t_data_tensor;

begin

  conv2d : entity work.conv2d
    port map (
      clk           => clk,
      fm_height     => fm_height,
      fm_width      => fm_width,
      input_tensor  => input_tensor,
      output_tensor => conv2d_output
      );

  relu : entity work.relu
    port map (
      clk           => clk,
      fm_height     => fm_height,
      fm_width      => fm_width,
      input_tensor  => conv2d_output,
      output_tensor => relu_output
      );
  batchnorm : entity work.batchnorm
    port map (
      clk           => clk,
      fm_height     => fm_height,
      fm_width      => fm_width,
      iter_index    => iter_index,
      input_tensor  => relu_output,
      output_tensor => batchnorm_output
      );

  residual_adder : entity work.residual_adder
    port map (
      clk           => clk,
      fm_height     => fm_height,
      fm_width      => fm_width,
      history       => history,
      input_tensor  => batchnorm_output,
      output_tensor => res_add_output
      );

  downsampler : entity work.downsampler
    port map (
      clk           => clk,
      enable        => ds_enable,
      fm_height     => fm_height,
      fm_width      => fm_width,
      input_tensor  => res_add_output,
      output_tensor => output_tensor
      );

end architecture rtl;
