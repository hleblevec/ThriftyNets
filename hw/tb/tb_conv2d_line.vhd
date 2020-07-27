
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library std;
use std.textio.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;
use work.pkg_csv_reader.all;
use work.pkg_tb_support.all;


entity tb_conv2d_line is
  generic(
    -- general
    CHECK_ERROR_LEVEL : severity_level := note;

    CLK_PERIOD      : time := 10.00 ns;
    CLK_EXTRA_DELAY : time := 0 ns;


    NUMBER_OF_BLOCKS : integer := 1
    );
  port(
    -- weights : out t_kernel_matrix;
    output_line_ref : out t_conv2d_feature_map_row
      );
end entity tb_conv2d_line;


architecture sim of tb_conv2d_line is

  -- ===========================================================================
  -- Signal declarations
  -- ===========================================================================
  signal enable_computation       : std_logic                               := '0';
  signal enable_reg               : std_logic                               := '0';
  signal data_available           : std_logic                               := '0';
  signal reset_accu               : std_logic                               := '0';
  signal reset_reg                : std_logic                               := '0';
  signal fm_width                 : natural range 0 to IN_FM_WIDTH          := IN_FM_WIDTH;
  signal output_channel_index     : natural range 0 to NUM_CHANNELS - 1     := 2;
  signal kernel_height_index      : integer range KERNEL_LOW to KERNEL_HIGH := KERNEL_LOW;
  signal kernel_channel_index     : natural range 0 to NUM_CHANNELS - 1     := 0;
  signal output_channel_index_reg : natural range 0 to NUM_CHANNELS - 1     := 0;
  signal kernel_height_index_reg  : integer range KERNEL_LOW to KERNEL_HIGH := KERNEL_LOW;
  signal kernel_channel_index_reg : natural range 0 to NUM_CHANNELS - 1     := 0;
  signal input_row                : t_conv2d_feature_map_row                := (others => (others => '0'));
  -- signal input_row_reg            : t_conv2d_feature_map_row                := (others => (others => '0'));
  signal output_row               : t_conv2d_feature_map_row                := (others => (others => '0'));


  signal row_index       : natural range 0 to IN_FM_HEIGHT + 1 := 3;
  -- signal output_line_ref : t_conv2d_feature_map_row            := (others => (others => '0'));

  signal fm_height_array : t_fm_height_array;
  signal conv2d_in       : t_conv2d_inputs;
  signal conv2d_out      : t_conv2d_outputs;

  -- Global
  signal clk        : std_logic := '0';
  signal rst        : std_logic := '1';
  signal test_count : integer   := 0;
  signal ref_count  : integer   := 0;

  -- signals for the clock generation
  constant CLK_HALF : time      := CLK_PERIOD / 2;
  signal clk_en     : std_logic := '1';

begin
  -- ===========================================================================
  -- Clock-Generator
  -- ===========================================================================
  clk <= clk nand clk_en after CLK_HALF;  -- change the frequency here


  -- ===========================================================================
  -- Read the configuration and the data files and apply stimuli
  -- ===========================================================================
  pr_data_loader : process
  -- variable fm_height_array : t_fm_height_array;
  begin
    rst <= '1';
    wait until clk = '1';

    wait for CLK_PERIOD;
    rst <= '0';

    -- ait for CLK_PERIOD;
    -- fm_heights_loop : for block_number in 0 to NUMBER_OF_BLOCKS-1 loop
    fm_height_array <= read_conv2d_fm_height_from_file(0);
    -- end loop;

    wait for CLK_PERIOD;
    input_loop : for block_number in 0 to NUMBER_OF_BLOCKS-1 loop
      conv2d_in  <= read_conv2d_in_from_file(block_number, fm_height_array);
      conv2d_out <= read_conv2d_out_from_file(block_number, fm_height_array);
    end loop;

    wait for CLK_PERIOD;

    loop_test : while test_count < NUM_TESTS loop
      -- report lf & "-------------------------" & lf & "-- Test #" & integer'image(test_count) & lf & "-------------------------";
      data_available           <= '1';
      fm_width                 <= fm_height_array(test_count);
      input_row                <= conv2d_in(test_count).input_tensor(kernel_channel_index)(row_index + kernel_height_index);
      -- input_row_reg            <= input_row;
      output_channel_index_reg <= output_channel_index;
      kernel_height_index_reg  <= kernel_height_index;
      kernel_channel_index_reg <= kernel_channel_index;
      enable_reg               <= enable_computation;
      reset_reg                <= reset_accu;
      wait for CLK_PERIOD;
    end loop;

    wait;                               -- forever, after data is loaded
  end process;

  test_count_incr : process
  begin
    wait until reset_accu = '1';
    wait for CLK_PERIOD;
    test_count <= test_count + 1;
  end process;


  -- ===========================================================================
  -- Test Outputs
  -- ===========================================================================
  pr_verif : process
    variable v_error_cnt   : integer;
    variable v_success_cnt : integer;
  begin
    v_error_cnt   := 0;
    v_success_cnt := 0;

    wait until clk = '1';
    wait for 20*CLK_PERIOD;             -- first stimuli are applied

    wait for CLK_HALF;

    loop_verif : while ref_count < NUM_TESTS loop
      wait until reset_accu = '1';
      wait for 2*CLK_PERIOD;

      assert output_row(0 to fm_height_array(ref_count)) = conv2d_out(ref_count).output_tensor(output_channel_index)(row_index)(0 to fm_height_array(ref_count)) report "Error in test #" & integer'image(ref_count) & "!" severity CHECK_ERROR_LEVEL;

      if output_row(0 to fm_height_array(ref_count)) /= conv2d_out(ref_count).output_tensor(output_channel_index)(row_index)(0 to fm_height_array(ref_count)) then
        v_error_cnt := v_error_cnt + 1;
      else
        v_success_cnt := v_success_cnt + 1;
      end if;
      output_line_ref <= conv2d_out(ref_count).output_tensor(output_channel_index)(row_index);
      ref_count <= ref_count + 1;
      wait for CLK_PERIOD;  -- delay of calculation in the recursion unit
    end loop;

    wait for 20*CLK_PERIOD;

    assert false
      report "Simulation of " & integer'image(v_success_cnt+v_error_cnt) & " test cases finished with " & integer'image(v_error_cnt) & " Errors."
      severity failure;

    wait;                               -- forever, after data is loaded
  end process;

  -- ===========================================================================
  -- Design under Test (DUT)
  -- ===========================================================================
  conv2d_compute : entity work.conv2d_compute
    port map (
      clk                  => clk,
      enable               => enable_reg,
      reset_accu           => reset_accu,
      fm_width             => fm_width,
      output_channel_index => output_channel_index_reg,
      kernel_height_index  => kernel_height_index_reg,
      kernel_channel_index => kernel_channel_index_reg,
      -- weights              => weights,
      input_row            => input_row,
      output_row           => output_row
      );

  conv2d_fsm : entity work.conv2d_fsm
    port map (
      clk                  => clk,
      data_available       => data_available,
      enable_computation   => enable_computation,
      reset_accu           => reset_accu,
      kernel_height_index  => kernel_height_index,
      kernel_channel_index => kernel_channel_index
      );


end architecture sim;
