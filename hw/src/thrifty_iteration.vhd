library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

use work.fixed_pkg.all;

entity thrifty_iteration is
  port(
    clk : in std_logic;

    fm_height : in natural range 1 to IN_FM_HEIGHT;
    fm_width  : in natural range 1 to IN_FM_WIDTH;

    iter_index : in natural range 0 to NUM_ITERATIONS - 1;
    ds_enable  : in std_logic;

    input_data_available  : in  std_logic;

    input_tensor  : in  t_data_tensor;
    output_tensor : out t_data_tensor

    );
end entity thrifty_iteration;

architecture rtl of thrifty_iteration is

  signal conv2d_input     : t_conv2d_data_tensor;
  signal conv2d_output    : t_conv2d_data_tensor;
  signal conv2d_output_available : std_logic;

  signal batchnorm_enable : std_logic;
  signal res_enable : std_logic;
  signal downsampler_enable : std_logic;

  signal relu_input       : t_data_tensor;
  signal relu_output      : t_data_tensor;
  signal batchnorm_output : t_data_tensor;
  signal res_add_intput   : t_data_tensor;
  signal res_add_output   : t_data_tensor;

  signal history : t_history;

  signal row_enable : std_logic_vector(IN_FM_HEIGHT - 1 downto 0);
  signal ds_row_enable : std_logic_vector(IN_FM_HEIGHT - 1 downto 0);

  function CUT_EDGES(conv2d_tensor : in t_conv2d_data_tensor) return t_data_tensor is
    variable data_tensor : t_data_tensor;
  begin
    for c in 0 to NUM_CHANNELS - 1 loop
      for h in 1 to IN_FM_HEIGHT loop
        for r in 1 to IN_FM_WIDTH loop
          data_tensor(c)(h-1)(r-1) := conv2d_tensor(c)(h)(r);
        end loop;
       end loop;
    end loop;
    return data_tensor;
  end function CUT_EDGES;

  function EXTRACT_HISTORY_ROWS(history : in t_history; channel_index :  in integer; height_index : in integer) return t_history_rows is
    variable history_rows :  t_history_rows;
  begin
    for i in 0 to HISTORY_DEPTH - 1 loop
      for r in 0 to IN_FM_WIDTH - 1 loop
        history_rows(i)(r) := history(i)(channel_index)(height_index)(r);
      end loop;
    end loop;
    return history_rows;
  end function EXTRACT_HISTORY_ROWS;


begin

  relu_input <= CUT_EDGES(conv2d_output);

  enable_reg : process(clk)
  begin
    if rising_edge(clk) then
      res_enable <= conv2d_output_available;
      batchnorm_enable <= res_enable;
      downsampler_enable <= batchnorm_enable and ds_enable;
    end if;
  end process;


  enable : process(fm_height)
  begin
    row_enable <= (fm_height - 1 downto 0 => '1', others => '0');
  end process;

  downsampling_rows : process(fm_height)
    begin
      for h in 0 to IN_FM_HEIGHT - 1 loop
        if h < to_integer(shift_right(to_unsigned(fm_height, 32), 2)) then
          ds_row_enable(2*h) <= '1';
          ds_row_enable(2*h + 1) <= '0';
        else
          ds_row_enable(h) <= '0';
        end if;
      end loop;
    end process;

  conv2d : entity work.conv2d
    port map (
      clk           => clk,
      fm_width      => fm_width,
      fm_height     => fm_height,
      input_data_available => input_data_available,
      output_data_available => conv2d_output_available,
      input_tensor  => conv2d_input,
      output_tensor => conv2d_output
      );

  channels : for c in 0 to NUM_CHANNELS - 1 generate
    rows : for h in 0 to IN_FM_HEIGHT - 1 generate
      signal local_relu_output_row : t_feature_map_row;
      signal local_batchnorm_output_row : t_feature_map_row;
      signal local_batchnorm_input_row : t_feature_map_row;
      signal local_res_output_row : t_feature_map_row;
      signal local_ds_input_row : t_feature_map_row;
      signal local_output_row : t_feature_map_row;
      signal local_history_rows : t_history_rows;

      signal local_batchnorm_enable : std_logic;
      signal local_res_enable : std_logic;
      signal local_downsampler_enable : std_logic;

    begin

      local_batchnorm_enable <= batchnorm_enable and row_enable(h);
      local_res_enable <= res_enable and row_enable(h);
      local_downsampler_enable <= ds_enable and ds_row_enable(h);
      local_history_rows <= EXTRACT_HISTORY_ROWS(history, c, h);

      
      local_batchnorm_input_row <= local_res_output_row;
      local_ds_input_row <= local_batchnorm_output_row;
     
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

      relu : entity work.relu
        port map (
          -- clk           => clk,
          -- enable        => relu_enable,
          fm_width   => fm_width,
          input_row  => relu_input(c)(h),
          output_row => local_relu_output_row
          );
      

      residual_adder : entity work.residual_adder
        port map (
          clk           => clk,
          enable        => local_res_enable,
          fm_width      => fm_width,
          history_rows  => local_history_rows,
          input_row  => local_relu_output_row,
          output_row => local_res_output_row
          );

        batchnorm : entity work.batchnorm
        port map (
          clk           => clk,
          enable        => local_batchnorm_enable,
          fm_width      => fm_width,
          iter_index    => iter_index,
          channel_index => c,
          row_index     => h,
          input_row  => local_batchnorm_input_row,
          output_row => local_batchnorm_output_row
          );

      downsampler : entity work.downsampler
        port map (
          clk           => clk,
          enable        => local_downsampler_enable,
          fm_width      => fm_width,
          input_row     => local_ds_input_row,
          output_row    => local_output_row
          );

    end generate;
  end generate;

end architecture rtl;
