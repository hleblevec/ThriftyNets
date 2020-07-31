library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity batchnorm is
  port(
    clk : in std_logic;

    enable : in std_logic;

    -- fm_height : in natural range 1 to IN_FM_HEIGHT;
    fm_width  : in natural range 1 to IN_FM_WIDTH;

    iter_index : in natural range 0 to NUM_ITERATIONS - 1;
    row_index : in natural range 0 to IN_FM_HEIGHT - 1;
    channel_index : in natural range 0 to NUM_CHANNELS - 1;

    input_row  : in  t_feature_map_row;
    output_row : out t_feature_map_row

    );
end entity batchnorm;

architecture rtl of batchnorm is

begin
  
  process(clk)
  begin
    for r in IN_FM_WIDTH - 1 loop
      if r < fm_width then
        if GAMMAS(iter_index)(channel_index)(row_index)(r) > 0 then:
          output_row(r) <= resize(SHIFT_LEFT(input_row(r), GAMMAS(iter_index)(channel_index)(row_index)(r)) - BIASES(iter_index)(channel_index), DATA_INT_BW - 1, -DATA_FRAC_BW);
        else
          output_row(r) <= resize(SHIFT_RIGHT(input_row(r), - GAMMAS(iter_index)(channel_index)(row_index)(r)) - BIASES(iter_index)(channel_index), DATA_INT_BW - 1, -DATA_FRAC_BW);
        end if;
      end if;
    end loop;
  end process;

end architecture rtl;
