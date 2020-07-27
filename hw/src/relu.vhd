library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity relu is
  port(
    clk : in std_logic;

    enable : in std_logic;

    -- fm_height : in natural range 1 to IN_FM_HEIGHT;
    fm_width : in natural range 1 to IN_FM_WIDTH;

    input_row  : in  t_feature_map_row;
    output_row : out t_feature_map_row

    );
end entity relu;

architecture rtl of relu is

begin

  process(clk)
  begin
    if rising_edge(clk) then
      for r in 0 to IN_FM_WIDTH - 1 loop
        if r < fm_width then
          if input_row(r)(DATA_INT_BW - 1) = '0' then
            output_row(r) <= input_row(r);
          else
            output_row(r) <= (others => '0');
          end if;
        end if;
      end loop;
    end if;
  end process;

end architecture rtl;
