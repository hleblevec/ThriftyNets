library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library work;
use work.pkg_param.all;
use work.pkg_types.all;
use work.pkg_lut.all;

entity global_pooling is
  port(
    clk : in std_logic;

    enable : in std_logic;

    -- fm_height : in natural range 1 to IN_FM_HEIGHT;
    fm_width : in natural range 1 to IN_FM_WIDTH;

    input_row  : in  t_feature_map_row;
    output_max : out t_data
    );
end entity global_pooling;

architecture rtl of global_pooling is

begin

  process(clk)
    variable max : t_data := (DATA_INT_BW - 1 => '1', others => '0');
  begin
    if rising_edge(clk) then
      for r in 0 to IN_FM_WIDTH - 1 loop
        if r < fm_width then
          if max < input_row(r) then
            max := input_row(r);
          end if;
        end if;
        output_max <= max;
        max        := (DATA_INT_BW - 1 => '1', others => '0');
      end loop;
    end if;
  end process;
end architecture rtl;
