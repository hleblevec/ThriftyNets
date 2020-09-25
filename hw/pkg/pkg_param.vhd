library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.math_real.all;

package pkg_param is

  constant NUM_CHANNELS   : natural := 16;
  constant NUM_CLASSES    : natural := 10;
  constant KERNEL_SIZE    : natural := 3;
  constant KERNEL_LOW     : integer := -1;
  constant KERNEL_HIGH    : integer := 1;
  constant NUM_ITERATIONS : natural := 7;
  constant IN_FM_HEIGHT   : natural := 28;
  constant IN_FM_WIDTH    : natural := 28;
  constant HISTORY_DEPTH  : natural := 1;

  constant DATA_INT_BW    : natural := 4;
  constant DATA_FRAC_BW   : natural := 12;
  constant WEIGHT_INT_BW  : natural := 1;
  constant WEIGHT_FRAC_BW : natural := 7;
  constant BIAS_INT_BW    : natural := 5;
  constant BIAS_FRAC_BW   : natural := 11;
  constant GAMMA_INT_BW   : natural := 8;
  constant GAMMA_FRAC_BW  : natural := 8;

  -- constant GAMMA_SHIFT_LOW : integer := 1;
  -- constant GAMMA_SHIFT_HIGH : integer := 6;

  constant ALPHA_0       : real    := 0.1;
  constant ALPHA_1       : real    := 0.9;
  constant ALPHA_FRAC_BW : natural := 8;

  constant POOLING_STRATEGY : std_logic_vector(NUM_ITERATIONS - 1 downto 0) := (others => '0');

  constant CONV2D_OVERFLOW : natural := 10;


end pkg_param;
