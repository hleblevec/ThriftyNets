
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


entity tb_batchnorm_line is
  generic(
    -- general
    CHECK_ERROR_LEVEL : severity_level := note;

    CLK_PERIOD      : time := 10.00 ns;
    CLK_EXTRA_DELAY : time := 0 ns;


    NUMBER_OF_BLOCKS : integer := 1
    );
  port(
    -- weights : out t_kernel_matrix;
    output_line_ref : out t_batchnorm_feature_map_row
    );
end entity tb_batchnorm_line;


architecture sim of tb_batchnorm_line is

  -- ===========================================================================
  -- Signal declarations
  -- ===========================================================================
  signal fm_width      : natural range 0 to IN_FM_WIDTH        := IN_FM_WIDTH;
  signal row_index     : natural range 0 to IN_FM_HEIGHT - 1   := 0;
  signal channel_index : natural range 0 to NUM_CHANNELS - 1   := 0;
  signal iter_index    : natural range 0 to NUM_ITERATIONS - 1 := 0;
  signal input_row     : t_batchnorm_feature_map_row           := (others => (others => '0'));
  signal output_row    : t_batchnorm_feature_map_row           := (others => (others => '0'));

  signal enable      : std_logic := '0';

  signal fm_height_array : t_fm_height_array;
  signal batchnorm_in    : t_batchnorm_inputs;
  signal batchnorm_out   : t_batchnorm_outputs;

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
      batchnorm_in  <= read_batchnorm_in_from_file(block_number, fm_height_array);
      batchnorm_out <= read_batchnorm_out_from_file(block_number, fm_height_array);
    end loop;

    wait for CLK_PERIOD;

    loop_test : while test_count < NUM_TESTS loop
      -- report lf & "-------------------------" & lf & "-- Test #" & integer'image(test_count) & lf & "-------------------------";
      enable           <= '1';
      fm_width                 <= fm_height_array(test_count);
      input_row                <= batchnorm_in(test_count).input_tensor(channel_index)(row_index);

      wait for CLK_PERIOD;
    end loop;

    wait;                               -- forever, after data is loaded
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
      wait for 2*CLK_PERIOD;

      assert output_row = batchnorm_out(ref_count).output_tensor(channel_index)(row_index) report "Error in test #" & integer'image(ref_count) & "!" severity CHECK_ERROR_LEVEL;

      if output_row /= batchnorm_out(ref_count).output_tensor(channel_index)(row_index) then
        v_error_cnt := v_error_cnt + 1;
      else
        v_success_cnt := v_success_cnt + 1;
      end if;
      output_line_ref <= batchnorm_out(ref_count).output_tensor(channel_index)(row_index);
      ref_count       <= ref_count + 1;
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

  DUT : entity work.batchnorm
    port map (
      clk           => clk,
      enable        => enable,
      fm_width      => fm_width,
      iter_index    => iter_index,
      row_index     => row_index,
      channel_index => channel_index,
      input_row     => input_row,
      output_row    => output_row
      );



end architecture sim;
