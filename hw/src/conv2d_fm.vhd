library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;
use work.fixed_pkg.all;

entity conv2d_fm is
  port(
    clk : in std_logic;

    input_data_available : in std_logic;
    output_data_available : out std_logic;

    output_channel_index : in natural range 0 to NUM_CHANNELS - 1;

    --fm_height : in natural range 0 to IN_FM_HEIGHT - 1;
    fm_width : in natural range 0 to IN_FM_WIDTH + 1;

    input_tensor  : in  t_conv2d_feature_map;
    output_tensor : out t_conv2d_feature_map
    );
end entity conv2d_fm;

architecture rtl of conv2d_fm is

  signal enable_computation   : std_logic;
  signal reset_accu           : std_logic;
  signal kernel_height_index  : integer range KERNEL_LOW to KERNEL_HIGH;
  signal kernel_channel_index : natural range 0 to NUM_CHANNELS - 1;


  signal enable_reg               : std_logic;
  signal kernel_height_index_reg  : integer range KERNEL_LOW to KERNEL_HIGH;
  signal kernel_channel_index_reg : natural range 0 to NUM_CHANNELS - 1;



begin

  output_data_available <= reset_accu;

  reg : process(clk)
  begin
    if rising_edge(clk) then
      enable_computation   <= enable_reg;
      kernel_height_index  <= kernel_height_index_reg;
      kernel_channel_index <= kernel_channel_index_reg;
    end if;
  end process;

    conv2d_row : for r in 1 to IN_FM_WIDTH generate
      signal local_enable               : std_logic;
      signal local_reset_accu           : std_logic;
      signal local_kernel_height_index  : integer range KERNEL_LOW to KERNEL_HIGH;
      signal local_kernel_channel_index : natural range 0 to NUM_CHANNELS - 1;
    begin

      global_update : if r = 1 generate
        enable_reg               <= local_enable;
        reset_accu               <= local_reset_accu;
        kernel_height_index_reg  <= local_kernel_height_index;
        kernel_channel_index_reg <= local_kernel_channel_index;
      end generate global_update;

      conv2d_compute : entity work.conv2d_compute
        port map (
          clk                  => clk,
          enable               => enable_computation,
          reset_accu           => reset_accu,
          fm_width             => fm_width,
          output_channel_index => output_channel_index,
          kernel_height_index  => kernel_height_index,
          kernel_channel_index => kernel_channel_index,
          input_row            => input_fm(r),
          output_row           => output_fm(r)
          );

      conv2d_fsm : entity work.conv2d_fsm
        port map (
          clk                  => clk,
          data_available       => input_data_available,
          enable_computation   => local_enable,
          reset_accu           => local_reset_accu,
          kernel_height_index  => local_kernel_height_index,
          kernel_channel_index => local_kernel_channel_index
          );
    end generate conv2d_row;


end architecture rtl;
