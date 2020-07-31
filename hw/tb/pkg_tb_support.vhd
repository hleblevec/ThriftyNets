library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.pkg_csv_reader.all;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;
use work.fixed_pkg.all;



package pkg_tb_support is

  constant NUM_TESTS : natural := 100;


  ------------------------------------------------------
  -- Types and functions for conv2d testbench --
  ------------------------------------------------------


  type trec_conv2d_in is record
    input_tensor : t_conv2d_data_tensor;
  end record;

  type trec_conv2d_out is record
    output_tensor : t_conv2d_data_tensor;
  end record;

  type t_fm_height_array is array(0 to NUM_TESTS-1) of natural range 0 to IN_FM_HEIGHT;
  type t_conv2d_inputs is array(0 to NUM_TESTS-1) of trec_conv2d_in;
  type t_conv2d_outputs is array(0 to NUM_TESTS-1) of trec_conv2d_out;

  impure function read_conv2d_fm_height_from_file(block_number : natural) return t_fm_height_array;
  impure function read_conv2d_in_from_file(block_number        : natural; fm_height : t_fm_height_array) return t_conv2d_inputs;
  impure function read_conv2d_out_from_file(block_number       : natural; fm_height : t_fm_height_array) return t_conv2d_outputs;

  ------------------------------------------------------
  -- Types and functions for batchnorm testbench --
  ------------------------------------------------------


  type trec_batchnorm_in is record
    input_tensor : t_data_tensor;
  end record;

  type trec_batchnorm_out is record
    output_tensor : t_data_tensor;
  end record;

  -- type t_fm_height_array is array(0 to NUM_TESTS-1) of natural range 0 to IN_FM_HEIGHT;
  type t_batchnorm_inputs is array(0 to NUM_TESTS-1) of trec_batchnorm_in;
  type t_batchnorm_outputs is array(0 to NUM_TESTS-1) of trec_batchnorm_out;

  -- impure function read_batchnorm_fm_height_from_file(block_number : natural) return t_fm_height_array;
  impure function read_batchnorm_in_from_file(block_number        : natural; fm_height : t_fm_height_array) return t_batchnorm_inputs;
  impure function read_batchnorm_out_from_file(block_number       : natural; fm_height : t_fm_height_array) return t_batchnorm_outputs;

end package pkg_tb_support;

package body pkg_tb_support is

  ----------------------------------------------
  -- ======================================== --
  -- == CONV2D UNIT TESTBENCH FUNCTIONS == --
  -- ======================================== --
  ----------------------------------------------

  impure function read_conv2d_fm_height_from_file(block_number : natural) return t_fm_height_array is
    variable v_csv   : csv_file_reader_type;
    variable v_input : t_fm_height_array;
  begin
    v_csv.initialize("test_data/N"& natural'image(NUM_TESTS) &"_b" & natural'image(block_number) & "_fm_height.csv");

    for i in 0 to NUM_TESTS-1 loop
      v_csv.readline;
      v_input(i) := v_csv.read_integer;

    end loop;

    return v_input;
  end function read_conv2d_fm_height_from_file;


  -- ===========================================================================
  -- FUNCTION: read_conv2d_unit_in_from_file
  -- Reads input data for the conv2d unit from files to
  -- t_conv2d_unit_inputs
  --
  --
  -- Data in the file is expected comma-separated, as follows:
  --
  -- Filename: "test_data/N<NUM_TESTS>_b<block_number>_conv2d_in.csv"
  --
  -- ===========================================================================

  impure function read_conv2d_in_from_file(block_number : natural; fm_height : t_fm_height_array) return t_conv2d_inputs is
    variable v_csv   : csv_file_reader_type;
    variable v_input : t_conv2d_inputs;
  begin
    v_csv.initialize("test_data/N"& natural'image(NUM_TESTS) &"_b" & natural'image(block_number) & "_conv2d_in.csv");

    for i in 0 to NUM_TESTS-1 loop

      for c in 0 to NUM_CHANNELS - 1 loop
        v_input(i).input_tensor(c)(0) := (others => (others => '0'));
        for h in 1 to IN_FM_HEIGHT + 1 loop
          if h < fm_height(i) + 1 then
            v_csv.readline;
            v_input(i).input_tensor(c)(h)(0) := (others => '0');
            for r in 1 to IN_FM_WIDTH + 1 loop
              if r < fm_height(i) + 1 then
                v_input(i).input_tensor(c)(h)(r) := to_sfixed(v_csv.read_real, DATA_INT_BW-1, -DATA_FRAC_BW);
              elsif r = fm_height(i) + 1 then
                v_input(i).input_tensor(c)(h)(r) := (others => '0');
              end if;
            end loop;
          elsif h = fm_height(i) + 1 then
            v_input(i).input_tensor(c)(h) := (others => (others => '0'));
          end if;
        end loop;
      end loop;

    end loop;

    return v_input;
  end function read_conv2d_in_from_file;



  -- ===========================================================================
  -- FUNCTION: read_conv2d_unit_out_from_file
  -- Reads input data for the conv2d unit from files to
  -- t_conv2d_unit_outputs
  --
  --
  -- Data in the file is expected comma-separated, as follows:
  --
  -- Filename: "test_data/N<NUM_TESTS>_b<block_number>_conv2d_out.csv"
  --
  -- ===========================================================================
  impure function read_conv2d_out_from_file(block_number : natural; fm_height : t_fm_height_array) return t_conv2d_outputs is
    variable v_csv    : csv_file_reader_type;
    variable v_output : t_conv2d_outputs;
  begin
    v_csv.initialize("test_data/N"& natural'image(NUM_TESTS) &"_b" & natural'image(block_number) & "_conv2d_out.csv");

    for i in 0 to NUM_TESTS-1 loop

      for c in 0 to NUM_CHANNELS - 1 loop
        v_output(i).output_tensor(c)(0) := (others => (others => '0'));
        for h in 1 to IN_FM_HEIGHT + 1 loop
          if h < fm_height(i) + 1 then
            v_csv.readline;
            v_output(i).output_tensor(c)(h)(0) := (others => '0');
            for r in 1 to IN_FM_WIDTH + 1 loop
              if r < fm_height(i) + 1 then
                v_output(i).output_tensor(c)(h)(r) := to_sfixed(v_csv.read_real, DATA_INT_BW-1, -DATA_FRAC_BW);
              else
                v_output(i).output_tensor(c)(h)(r) := (others => '0');
              end if;
            end loop;
          else
            v_output(i).output_tensor(c)(h) := (others => (others => '0'));
          end if;
        end loop;
      end loop;

    end loop;

    return v_output;
  end function read_conv2d_out_from_file;


  ----------------------------------------------
  -- ======================================== --
  -- == BATCHNORM UNIT TESTBENCH FUNCTIONS == --
  -- ======================================== --
  ----------------------------------------------

  -- ===========================================================================
  -- FUNCTION: read_batchnorm_unit_in_from_file
  -- Reads input data for the batchnorm unit from files to
  -- t_batchnorm_unit_inputs
  --
  --
  -- Data in the file is expected comma-separated, as follows:
  --
  -- Filename: "test_data/N<NUM_TESTS>_b<block_number>_batchnorm_in.csv"
  --
  -- ===========================================================================

  impure function read_batchnorm_in_from_file(block_number : natural; fm_height : t_fm_height_array) return t_batchnorm_inputs is
    variable v_csv   : csv_file_reader_type;
    variable v_input : t_batchnorm_inputs;
  begin
    v_csv.initialize("test_data/N"& natural'image(NUM_TESTS) &"_b" & natural'image(block_number) & "_batchnorm_in.csv");

    for i in 0 to NUM_TESTS-1 loop

      for c in 0 to NUM_CHANNELS - 1 loop
        for h in 0 to IN_FM_HEIGHT - 1 loop
          if h < fm_height(i) then
            v_csv.readline;
            for r in 0 to IN_FM_WIDTH - 1 loop
              if r < fm_height(i) then
                v_input(i).input_tensor(c)(h)(r) := to_sfixed(v_csv.read_real, DATA_INT_BW-1, -DATA_FRAC_BW);
              end if;
            end loop;
          end if;
        end loop;
      end loop;

    end loop;

    return v_input;
  end function read_batchnorm_in_from_file;



  -- ===========================================================================
  -- FUNCTION: read_batchnorm_unit_out_from_file
  -- Reads input data for the batchnorm unit from files to
  -- t_batchnorm_unit_outputs
  --
  --
  -- Data in the file is expected comma-separated, as follows:
  --
  -- Filename: "test_data/N<NUM_TESTS>_b<block_number>_batchnorm_out.csv"
  --
  -- ===========================================================================
  impure function read_batchnorm_out_from_file(block_number : natural; fm_height : t_fm_height_array) return t_batchnorm_outputs is
    variable v_csv    : csv_file_reader_type;
    variable v_output : t_batchnorm_outputs;
  begin
    v_csv.initialize("test_data/N"& natural'image(NUM_TESTS) &"_b" & natural'image(block_number) & "_batchnorm_out.csv");

    for i in 0 to NUM_TESTS-1 loop

      for c in 0 to NUM_CHANNELS - 1 loop
        for h in 0 to IN_FM_HEIGHT - 1 loop
          if h < fm_height(i) then
            v_csv.readline;
            for r in 0 to IN_FM_WIDTH - 1 loop
              if r < fm_height(i) then
                v_output(i).output_tensor(c)(h)(r) := to_sfixed(v_csv.read_real, DATA_INT_BW-1, -DATA_FRAC_BW);
              end if;
            end loop;
          end if;
        end loop;
      end loop;

    end loop;

    return v_output;
  end function read_batchnorm_out_from_file;




end package body pkg_tb_support;
