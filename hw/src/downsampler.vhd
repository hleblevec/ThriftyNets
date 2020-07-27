library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.math_real.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity downsampler is
  port(
    clk : in std_logic;

    enable : in std_logic;

    -- fm_height : in natural range 0 to IN_FM_HEIGHT;
    fm_width  : in natural range 0 to IN_FM_WIDTH;

    input_row : in t_data_tensor;
    output_row : out t_data_tensor

  );
end entity downsampler;

architecture rtl of downsampler is

  constant FM_WIDTH_BW : natural := to_integer(ceil(log2(real(IN_FM_WIDTH))));
  signal out_fm_width : natural range 0 to IN_FM_WIDTH;

  begin

    out_fm_width <= to_integer(shift_right(to_unsigned(fm_width, FM_WIDTH_BW), 2));

    process(clk)
    begin
      if rising_edge(clk) then
        for r in 0 to IN_FM_WIDTH - 1 loop
          if r < out_fm_width then
            output_row(r) <= input_row(2*r);
          end if;
        end loop;
      end if;
    end process;

end architecture rtl;
