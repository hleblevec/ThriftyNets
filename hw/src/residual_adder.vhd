library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity residual_adder is
  port(
    clk : in std_logic;

    enable : in std_logic;

    -- fm_height : in integer;
    fm_width : in integer;

    history_rows : in t_history;

    input_row  : in  t_feature_map_row;
    output_row : out t_feature_map_row

    );
end entity residual_adder;

architecture rtl of residual_adder is

  constant ALPHA_0_FIXED : ufixed((ALPHA_FRAC_BW + 1) - 1 downto 0) := to_ufixed(ALPHA_0, 1, ALPHA_FRAC_BW);
  constant ALPHA_1_FIXED : ufixed((ALPHA_FRAC_BW + 1) - 1 downto 0) := to_ufixed(ALPHA_1, 1, ALPHA_FRAC_BW);

begin

  process(clk)
    variable sum : sfixed((DATA_INT_BW + HISTORY_DEPTH + 2) - 1 downto - DATA_FRAC_BW) := (others => '0');
  begin
    if rising_edge(clk) then
      for r in IN_FM_WIDTH - 1 loop
        if r < fm_width then
          for i in 0 to HISTORY_DEPTH - 1 loop
            sum := sum + ALPHA_1_FIXED * history(i)(r);
          end loop;
          output_row(r) <= resize(sum + ALPHA_0_FIXED * input_row(r), DATA_INT_BW, DATA_FRAC_BW);
          sum           := (others => '0');
        end if;
      end loop;
    end if;
  end process;

end architecture rtl;
