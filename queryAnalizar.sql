select
    stat_date,
    a.shop_id,
    CASE
        WHEN stat_date < DATE('2024-04-01') THEN r_performance
        WHEN r_performance_2 is not null then r_performance_2
        ELSE 'R3'
    END as r_performance
from
    (
        SELECT
            concat_ws('-', year, month, day) as stat_date,
            CONCAT_WS(
                '-',
                '2024',
                MONTH(concat_ws('-', year, month, day))
            ) as month_,
            shop_id,
            r_performance
        from
            soda_international_dwm.dwm_shop_wide_d_whole
        where
            concat_ws('-', year, month, day) between '2024-01-01'
            and '2024-06-30'
            AND country_code in('MX', 'CO', 'CR', 'PE')
    ) as a
    left join (
        SELECT
            date_,
            shop_id,
            r_performance as r_performance_2
        from
            dp_det_data.old_r_performance
        where
            date_ between '2024-01-01'
            and '2024-06-30'
            AND country_code in('MX', 'CO', 'CR', 'PE')
    ) as b on a.shop_id = b.shop_id
    and a.month_ = b.date_