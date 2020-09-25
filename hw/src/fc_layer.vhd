library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

use work.fixed_pkg.all;

entity fc_layer is
  port(
    clk : in std_logic;

    fc_input  : in  t_fc_input;
    fc_output : out t_fc_output
    );
end entity fc_layer;

architecture rtl of fc_layer is

begin
  process(clk)
    variable sum : sfixed((DATA_INT_BW + WEIGHT_INT_BW + 1) - 1 downto - (DATA_FRAC_BW + WEIGHT_FRAC_BW)) := (others => '0');
  begin
    if rising_edge(clk) then
      for n in 0 to NUM_CLASSES - 1 loop
        for c in 0 to NUM_CHANNELS - 1 loop
          sum := sum + fc_input(c) * FC_WEIGHTS(n)(c);
        end loop;
        fc_output(n) <= resize(sum, DATA_INT_BW, DATA_FRAC_BW);
        sum := (others => '0')
      end loop;
    end if;
  end process;

end architecture;
