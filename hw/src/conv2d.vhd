library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;
use work.fixed_pkg.all;

entity conv2d is
  port(
    clk : in std_logic;

    input_data_available  : in  std_logic;
    output_data_available : out std_logic;

    fm_height : in natural range 0 to IN_FM_HEIGHT + 1;
    fm_width : in natural range 0 to IN_FM_WIDTH + 1;

    input_tensor  : in  t_conv2d_data_tensor;
    output_tensor : out t_conv2d_data_tensor
    );
end entity conv2d;

architecture rtl of conv2d is

  signal enable_computation   : std_logic;
  signal reset_accu           : std_logic;
  signal kernel_height_index  : integer range KERNEL_LOW to KERNEL_HIGH;
  signal kernel_channel_index : natural range 0 to NUM_CHANNELS - 1;


  signal enable_reg               : std_logic;
  signal kernel_height_index_reg  : integer range KERNEL_LOW to KERNEL_HIGH;
  signal kernel_channel_index_reg : natural range 0 to NUM_CHANNELS - 1;

  signal output_tensor_reg : t_conv2d_data_tensor;

  signal row_enable : std_logic_vector(IN_FM_HEIGHT downto 0);



begin

  enable : process(fm_height)
  begin
    row_enable <= (fm_height downto 1 => '1', others => '0');
  end process;

  reg : process(clk)
  begin
    if rising_edge(clk) then
      enable_computation    <= enable_reg;
      kernel_height_index   <= kernel_height_index_reg;
      kernel_channel_index  <= kernel_channel_index_reg;
      output_data_available <= reset_accu;
    end if;
  end process;

  conv2d_channel : for c in 0 to NUM_CHANNELS - 1 generate

    null_borders : process(clk)
    begin
      if rising_edge(clk) then
        output_tensor(c)(0)            <= (others => (others => '0'));
        output_tensor(c)(IN_FM_HEIGHT + 1) <= (others => (others => '0'));
      end if;
    end process;

    conv2d_row : for h in 1 to IN_FM_HEIGHT generate
      signal local_fsm_enable           : std_logic;
      signal local_enable_computation   : std_logic;
      signal local_enable               : std_logic;
      signal local_reset_accu           : std_logic;
      signal local_kernel_height_index  : integer range KERNEL_LOW to KERNEL_HIGH;
      signal local_kernel_channel_index : natural range 0 to NUM_CHANNELS - 1;
      signal local_output_row           : t_conv2d_feature_map_row;
    begin

      global_update : if c = 0 and h = 1 generate
        enable_reg               <= local_enable;
        reset_accu               <= local_reset_accu;
        kernel_height_index_reg  <= local_kernel_height_index;
        kernel_channel_index_reg <= local_kernel_channel_index;
      end generate global_update;

      local_fsm_enable <= input_data_available and row_enable(h);
      local_enable_computation <= enable_computation and row_enable(h);

      write_output : process(clk)
      begin
        if rising_edge(clk) then
          if h < fm_width + 1 then
            output_tensor(c)(h) <= local_output_row;
          else
            output_tensor(c)(h) <= (others => (others => '0'));
          end if;
        end if;
      end process;

      conv2d_compute : entity work.conv2d_compute
        port map (
          clk                  => clk,
          enable               => local_enable_computation,
          reset_accu           => reset_accu,
          fm_width             => fm_width,
          output_channel_index => c,
          kernel_height_index  => kernel_height_index,
          kernel_channel_index => kernel_channel_index,
          input_row            => input_tensor(kernel_channel_index)(h + kernel_height_index),
          output_row           => local_output_row
          );

      conv2d_fsm : entity work.conv2d_fsm
        port map (
          clk                  => clk,
          data_available       => local_fsm_enable,
          enable_computation   => local_enable,
          reset_accu           => local_reset_accu,
          kernel_height_index  => local_kernel_height_index,
          kernel_channel_index => local_kernel_channel_index
          );
    end generate conv2d_row;
  end generate conv2d_channel;


end architecture rtl;
