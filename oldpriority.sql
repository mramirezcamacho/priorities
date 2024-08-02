SELECT
    country_code,
    MONTH(stat_date) as month_,
    priority,
    sum(is_effective_online) / COUNT(DISTINCT stat_date) as eff_online_rs,
    sum(complete_orders) as complete_orders,
    sum(complete_orders) / COUNT(DISTINCT stat_date) as daily_orders,
    sum(complete_orders) / sum(is_effective_online) as orders_per_eff_online,
    SUM(is_active) / COUNT(DISTINCT stat_date) as active_rs,
    sum(b_cancel_order_cnt) / sum(pay_order_cnt) as b_cancel_rate,
    sum(bad_rating_order_cnt_b) / SUM(complete_orders) as bad_rating_rate,
    sum(horas_abiertas) / COUNT(DISTINCT stat_date) as daily_online_hours,
    sum(is_healthy_store) / COUNT(DISTINCT stat_date) as healthy_stores,
    sum(shop_exp_user_cnt_app) as exposure_uv,
    sum(shop_enter_user_cnt_app) as shop_enter_uv,
    sum(click_pay_user_cnt_app) as click_pay_user_cnt_app,
    sum(pay_success_user_cnt_app) as pay_success_user_cnt_app,
    sum(shop_enter_user_cnt_app) / sum(shop_exp_user_cnt_app) as b_p1,
    sum(click_pay_user_cnt_app) / sum(shop_enter_user_cnt_app) as b_p2,
    sum(click_pay_user_cnt_app) / sum(shop_exp_user_cnt_app) as b_p1p2,
    SUM(r_burn) as r_burn,
    SUM(r_burn) / SUM(gmv) as r_burn_gmv,
    SUM(r_burn) / SUM(complete_orders) as r_burn_per_order,
    SUM(b2c_total_burn) as b2c_total_burn,
    SUM(b2c_total_burn) / SUM(gmv) as b2c_gmv,
    SUM(b2c_total_burn) / SUM(complete_orders) as b2c_per_order,
    sum(commission) / sum(complete_orders) as tri_per_order,
    SUM(total_p2c_burn) as total_p2c_burn,
    SUM(total_p2c_burn) / SUM(gmv) as p2c_gmv,
    SUM(total_p2c_burn) / SUM(complete_orders) as p2c_burn_per_order,
    SUM(ted) as ted,
    SUM(ted) / SUM(gmv) as ted_gmv,
    sum(ted) / sum(complete_orders) as ted_per_order,
    SUM(imperfect_orders) as imperfect_orders,
    SUM(imperfect_orders) / SUM(total_orders) as imperfect_order_rate,
    sum(pay_order_cnt) as pay_order_cnt,
    sum(b_cancel_order_cnt) as b_cancel_order_cnt,
    sum(shop_exp_user_cnt_app) / sum(is_effective_online) as exposure_per_eff_online
FROM
    (
        SELECT
            country_code,
            shop_basic.stat_date,
            --priority,
            shop_basic.shop_id,
            --priority.potential,
            CASE
                --#! here starts the new code for is_new
                WHEN lifecycle_type = 'New Rs' THEN 'Priority 1' --#! here ends the new code for is_new
                WHEN r_performance = 'R1' THEN 'Priority 2'
                WHEN potential = 'T1'
                AND r_performance = 'R3' THEN 'Priority 3'
                WHEN r_performance = 'R2' THEN 'Priority 4'
                ELSE 'Priority 5'
            END AS priority,
            is_effective_online,
            complete_orders,
            is_active,
            b_cancel_order_cnt,
            pay_order_cnt,
            bad_rating_order_cnt_b,
            horas_abiertas,
            is_healthy_store,
            shop_exp_user_cnt_app,
            shop_enter_user_cnt_app,
            click_pay_user_cnt_app,
            pay_success_user_cnt_app,
            r_burn,
            total_orders,
            gmv,
            b2c_total_burn,
            commission,
            total_p2c_burn,
            ted,
            imperfect_orders
        FROM
            (
                SELECT
                    concat_ws('-', year, month, day) as stat_date,
                    country_code,
                    shop_id,
                    ka_type
                FROM
                    soda_international_dwm.dwm_bizopp_wide_d_whole
                where
                    country_code in ('MX', 'CO', 'CR', 'PE')
                    and concat_ws('-', year, month, day) between '2024-01-01'
                    AND '2024-06-30'
                    and organization_type = 4
                    and business_type = 1
            ) as shop_basic --#! here starts the new code for is_new
            LEFT JOIN(
                select
                    concat_ws('-', year, month, day) as stat_date,
                    shop_id,
                    CASE
                        WHEN datediff(
                            DATE(concat_ws('-', year, month, day)),
                            DATE(substr(first_online_time, 1, 10))
                        ) + 1 > 60 THEN 'Mature'
                        WHEN datediff(
                            DATE(concat_ws('-', year, month, day)),
                            DATE(substr(first_online_time, 1, 10))
                        ) + 1 <= 60 THEN 'New Rs'
                        ELSE 'No First Online'
                    END as lifecycle_type
                from
                    soda_international_dwm.dwm_bizopp_sign_process_d_whole
                where
                    country_code in ('MX', 'CO', 'PE', 'CR')
                    and first_online_time is not null
                    and concat_ws('-', year, month, day) between '2024-01-01'
                    and '2024-06-30'
            ) as first_online on shop_basic.shop_id = first_online.shop_id
            and shop_basic.stat_date = first_online.stat_date
            LEFT JOIN (
                SELECT
                    shop_id,
                    --priority,
                    case
                        when potential = '1' then 'T1'
                        when potential = '2' then 'T2'
                        when potential = '3' then 'T3'
                        else ''
                    end as potential
                FROM
                    soda_international_dwm.dwm_bizopp_wide_d_whole
                where
                    country_code in ('MX', 'CO', 'CR', 'PE')
                    and concat_ws('-', year, month, day) = '2024-06-30'
                    and organization_type = 4
                    and business_type = 1
            ) as priority on shop_basic.shop_id = priority.shop_id
            left join (
                SELECT
                    stat_date,
                    shop_id,
                    r_performance
                FROM
                    (
                        SELECT
                            concat_ws('-', year, month, day) as stat_date,
                            shop_id,
                            r_performance
                        from
                            soda_international_dwm.dwm_shop_wide_d_whole
                        where
                            concat_ws('-', year, month, day) between '2024-01-01'
                            and '2024-03-31'
                            AND country_code in('MX', 'CO', 'CR', 'PE')
                        union all
                        SELECT
                            stat_date,
                            shop.shop_id,
                            CASE
                                WHEN r_performance is not null then r_performance
                                ELSE 'R3'
                            END as r_performance
                        FROM
                            (
                                SELECT
                                    concat_ws('-', year, month, day) as stat_date,
                                    shop_id
                                from
                                    soda_international_dwm.dwm_shop_wide_d_whole
                                where
                                    concat_ws('-', year, month, day) between '2024-04-01'
                                    and '2024-04-30'
                                    AND country_code in('MX', 'CO', 'CR', 'PE')
                            ) as shop
                            left join(
                                select
                                    shop_id,
                                    CASE
                                        WHEN SUM(complete_order_num) / COUNT(DISTINCT concat_ws('-', year, month, day)) >= 5 THEN 'R1'
                                        WHEN SUM(complete_order_num) / COUNT(DISTINCT concat_ws('-', year, month, day)) > 1 THEN 'R2'
                                        ELSE 'R3'
                                    END as r_performance
                                from
                                    soda_international_dwm.dwm_shop_wide_d_whole
                                where
                                    concat_ws('-', year, month, day) between '2024-03-01'
                                    and '2024-03-31'
                                    AND country_code in('MX', 'CO', 'CR', 'PE')
                                group by
                                    shop_id
                            ) as r_performance_apr on shop.shop_id = r_performance_apr.shop_id
                        union all
                        SELECT
                            stat_date,
                            shop.shop_id,
                            CASE
                                WHEN r_performance is not null then r_performance
                                ELSE 'R3'
                            END as r_performance
                        FROM
                            (
                                SELECT
                                    concat_ws('-', year, month, day) as stat_date,
                                    shop_id
                                from
                                    soda_international_dwm.dwm_shop_wide_d_whole
                                where
                                    concat_ws('-', year, month, day) between '2024-05-01'
                                    and '2024-05-31'
                                    AND country_code in('MX', 'CO', 'CR', 'PE')
                            ) as shop
                            left join(
                                select
                                    shop_id,
                                    CASE
                                        WHEN SUM(complete_order_num) / COUNT(DISTINCT concat_ws('-', year, month, day)) >= 5 THEN 'R1'
                                        WHEN SUM(complete_order_num) / COUNT(DISTINCT concat_ws('-', year, month, day)) > 1 THEN 'R2'
                                        ELSE 'R3'
                                    END as r_performance
                                from
                                    soda_international_dwm.dwm_shop_wide_d_whole
                                where
                                    concat_ws('-', year, month, day) between '2024-04-01'
                                    and '2024-04-30'
                                    AND country_code in('MX', 'CO', 'CR', 'PE')
                                group by
                                    shop_id
                            ) as r_performance_may on shop.shop_id = r_performance_may.shop_id
                        union all
                        SELECT
                            stat_date,
                            shop.shop_id,
                            CASE
                                WHEN r_performance is not null then r_performance
                                ELSE 'R3'
                            END as r_performance
                        FROM
                            (
                                SELECT
                                    concat_ws('-', year, month, day) as stat_date,
                                    shop_id
                                from
                                    soda_international_dwm.dwm_shop_wide_d_whole
                                where
                                    concat_ws('-', year, month, day) between '2024-06-01'
                                    and '2024-06-30'
                                    AND country_code in('MX', 'CO', 'CR', 'PE')
                            ) as shop
                            left join(
                                select
                                    shop_id,
                                    CASE
                                        WHEN SUM(complete_order_num) / COUNT(DISTINCT concat_ws('-', year, month, day)) >= 5 THEN 'R1'
                                        WHEN SUM(complete_order_num) / COUNT(DISTINCT concat_ws('-', year, month, day)) > 1 THEN 'R2'
                                        ELSE 'R3'
                                    END as r_performance
                                from
                                    soda_international_dwm.dwm_shop_wide_d_whole
                                where
                                    concat_ws('-', year, month, day) between '2024-05-01'
                                    and '2024-05-31'
                                    AND country_code in('MX', 'CO', 'CR', 'PE')
                                group by
                                    shop_id
                            ) as r_performance_jun on shop.shop_id = r_performance_jun.shop_id
                    ) as r_performance_joint
            ) as r_performance on shop_basic.shop_id = r_performance.shop_id
            and shop_basic.stat_date = r_performance.stat_date
            left join (
                select
                    concat_ws('-', year, month, day) as stat_date,
                    shop_id,
                    nvl(is_effective_online, 0) as is_effective_online,
                    nvl(set_duration, 0) / 3600 as horas_disponibles,
                    nvl(open_duration_in_set, 0) / 3600 as horas_abiertas,
                    CASE
                        WHEN open_duration_in_set / set_duration > 0.6
                        and is_headimg_default = 0
                        and is_headimg = 1
                        and available_item_num >= 5
                        and is_logoimg = 1
                        and is_logoimg_default = 0
                        and available_pic_item_num / available_item_num >= 0.6 THEN 1
                        ELSE 0
                    END as is_healthy_store,
                    nvl(complete_order_num, 0) as complete_orders,
                    if(complete_order_num > 0, 1, 0) as is_active,
                    nvl(bad_rating_order_cnt_b, 0) as bad_rating_order_cnt_b,
                    nvl(bad_rating_order_cnt_d, 0) as bad_rating_order_cnt_d,
                    nvl(b_cancel_order_cnt, 0) as b_cancel_order_cnt,
                    nvl(c_cancel_order_cnt, 0) as c_cancel_order_cnt,
                    nvl(d_cancel_order_cnt, 0) as d_cancel_order_cnt,
                    nvl(p_cancel_order_cnt, 0) as p_cancel_order_cnt,
                    nvl(pay_order_cnt, 0) as pay_order_cnt
                from
                    soda_international_dwm.dwm_shop_wide_d_whole
                where
                    concat_ws('-', year, month, day) between '2024-01-01'
                    and '2024-06-30'
                    AND country_code in('MX', 'CO', 'CR', 'PE')
            ) as shop_active on shop_basic.stat_date = shop_active.stat_date
            and shop_basic.shop_id = shop_active.shop_id
            left join (
                SELECT
                    concat_ws('-', year, month, day) as stat_date,
                    shop_id,
                    shop_exp_user_cnt_app,
                    shop_enter_user_cnt_app,
                    click_pay_user_cnt_app,
                    pay_success_user_cnt_app
                FROM
                    soda_international_dwm.dwm_traffic_shop_convert_d_whole
                WHERE
                    concat_ws('-', year, month, day) between '2024-01-01'
                    and '2024-06-30'
                    and country_code in ('MX', 'CO', 'CR', 'PE')
            ) as experience on shop_basic.stat_date = experience.stat_date
            and shop_basic.shop_id = experience.shop_id
            left join (
                SELECT
                    concat_ws('-', year, month, day) as stat_date,
                    shop_id,
                    gmv,
                    realpay_price as realpay_price,
                    r_burn as r_burn,
                    (b2c_acq_burn + b2c_eng_burn) as b2c_total_burn,
                    commission,
                    (p2c_eng_burn + p2c_acq_burn) as total_p2c_burn,
                    ted
                FROM
                    soda_international_dwm.dwm_finance_byshop_d_increment
                WHERE
                    concat_ws('-', year, month, day) between '2024-01-01'
                    and '2024-06-30'
                    and country_code in ('MX', 'CO', 'CR', 'PE')
            ) as finance on shop_basic.stat_date = finance.stat_date
            AND shop_basic.shop_id = finance.shop_id
            LEFT JOIN (
                select
                    order_base.stat_date,
                    order_base.shop_id,
                    count(
                        distinct case
                            when nvl(missing_food, 0) + nvl(Wrong_Food, 0) + nvl(Whole_Sent_Wrong, 0) + nvl(missing_food_wrong_food_whole_sent_wrong_cpo, 0) + nvl(c_d_tag_ids, 0) > 0 then order_base.order_id
                        end
                    ) as imperfect_orders,
                    count(
                        distinct case
                            when is_td_complete = 1 then order_base.order_id
                        end
                    ) as total_orders
                from
                    (
                        select
                            concat_ws('-', year, month, day) as stat_date,
                            shop_id,
                            concat("_", city_id) as city_id,
                            upper(country_code) as country_code,
                            is_td_complete,
                            is_td_pay,
                            concat('_', order_id) as order_id
                        from
                            soda_international_dwd.dwd_order_wide_d_increment
                        where
                            1 = 1
                            and upper(country_code) in ('MX', 'CO', 'CR', 'PE')
                            and concat_ws('-', year, month, day) between '2024-01-01'
                            AND '2024-06-30'
                            and channel = 0
                            and (
                                '0' == '0'
                                or city_id in (0)
                            )
                            and (
                                is_td_complete = 1
                                or is_td_cancel = 1
                                or is_td_pay = 1
                            )
                    ) as order_base
                    LEFT JOIN (
                        select
                            --stat_date,
                            shop_id,
                            order_id,
                            c_d_tag_ids
                        from
                            (
                                select
                                    --order_date as stat_date,
                                    shop_id,
                                    order_id,
                                    case
                                        when (
                                            c_d_tag_ids like '%302011%'
                                            or c_b_tag_ids like '%202001%'
                                            or c_b_tag_ids like '%202003%'
                                            or c_b_tag_ids like '%202006%'
                                            or c_b_tag_ids like '%202014%'
                                        ) then 1
                                        else 0
                                    end as c_d_tag_ids
                                from
                                    (
                                        select
                                            shop_id,
                                            order_id,
                                            order_date,
                                            c_d_tag_ids,
                                            c_b_tag_ids,
                                            regexp_replace(sub_comment, '["\\[\\]]', '') as user_comment
                                        from
                                            (
                                                SELECT
                                                    to_date(complete_time_local) as order_date,
                                                    shop_id,
                                                    concat('_', order_id) as order_id,
                                                    c_b_tags AS categories,
                                                    c_b_score as score,
                                                    c_b_tag_ids,
                                                    c_d_tag_ids
                                                FROM
                                                    soda_international_dwd.dwd_order_evaluation_d_increment AS evals
                                                WHERE
                                                    1 = 1
                                                    AND country_code in ('MX', 'CO', 'CR', 'PE')
                                                    and concat_ws('-', year, month, day) between '2024-01-01'
                                                    AND '2024-06-30'
                                            ) as tb lateral view explode(split(categories, ',')) Subs AS sub_comment
                                    ) b
                            ) c
                    ) as bad_reviews on order_base.order_id = bad_reviews.order_id --and order_base.stat_date = bad_reviews.stat_date
                    LEFT JOIN (
                        select
                            *
                        from
                            (
                                select
                                    --concat_ws('-', year, month, day) as stat_date,
                                    country_code,
                                    apply_id,
                                    concat('_', order_id) as order_id,
                                    apply_type,
                                    split(selected_reason_tag_list, ',') as reason_tag_list,
                                    case
                                        when array_CONTAINS(split(selected_reason_tag_list, ','), '1') then 1
                                        else 0
                                    end as Missing_Food,
                                    case
                                        when array_contains(split(selected_reason_tag_list, ','), '2')
                                        and (
                                            get_json_object(
                                                get_json_object(apply_info, '$.additionalDetails'),
                                                '$.wholeSentWrong'
                                            ) is null
                                            or get_json_object(
                                                get_json_object(apply_info, '$.additionalDetails'),
                                                '$.wholeSentWrong'
                                            ) = 'false'
                                        ) then 1
                                        else 0
                                    end as Wrong_Food,
                                    case
                                        when array_CONTAINS(split(selected_reason_tag_list, ','), '3') then 1
                                        else 0
                                    end as Food_Damage,
                                    case
                                        when array_CONTAINS(split(selected_reason_tag_list, ','), '6') then 1
                                        else 0
                                    end as Food_Quality,
                                    case
                                        when apply_type = 4 then 1
                                        else 0
                                    end as Didnt_Receive_Order,
                                    case
                                        when array_contains(split(selected_reason_tag_list, ','), '2')
                                        and get_json_object(
                                            get_json_object(apply_info, '$.additionalDetails'),
                                            '$.wholeSentWrong'
                                        ) = 'true' then 1
                                        else 0
                                    end as Whole_Sent_Wrong,
                                    row_number() over(
                                        partition by apply_id,
                                        order_id
                                        order by
                                            update_time_local desc
                                    ) rn,
                                    get_json_object(
                                        get_json_object(apply_info, '$.additionalDetails'),
                                        '$.sealDetail'
                                    ) as seal_detail
                                from
                                    soda_international_dwd.dwd_order_refund_apply_d_increment
                                where
                                    1 = 1
                                    and concat_ws('-', year, month, day) between '2024-01-01'
                                    AND '2024-06-30'
                                    and apply_type = 3
                            ) m
                        where
                            rn = 1
                            and country_code in ('MX', 'CO', 'CR', 'PE')
                    ) as after_sales on after_sales.order_id = order_base.order_id --and order_base.stat_date = after_sales.stat_date
                    LEFT JOIN (
                        select
                            concat('_', order_id) as order_id,
                            --concat_ws('-', year, month, day) as stat_dt,
                            category_id,
                            case
                                when category_id in (
                                    '960030884',
                                    '960020854',
                                    '960030886',
                                    '960020822',
                                    '960030876',
                                    '960030890',
                                    '960020860',
                                    '960030892',
                                    '960105621',
                                    '960105619',
                                    '960020862',
                                    '960030894',
                                    '960020856',
                                    '960030888',
                                    '960020824'
                                ) then 1
                                else 0
                            end as missing_food_wrong_food_whole_sent_wrong_cpo,
                            case
                                when category_id in ('960020854', '960030886') then 1
                                else 0
                            end as small_portions_c,
                            case
                                when category_id in ('960030884') then 1
                                else 0
                            end as damaged_c,
                            case
                                when category_id in ('960020862', '960030894') then 1
                                else 0
                            end as wrong_c,
                            case
                                when category_id in ('960020856', '960030888') then 1
                                else 0
                            end as missing_c,
                            case
                                when category_id in ('960020860', '960030892') then 1
                                else 0
                            end as different_c,
                            case
                                when category_id in (
                                    '960020822',
                                    '960030876',
                                    '960030890',
                                    '960105621',
                                    '960105619',
                                    '960020824'
                                ) then 1
                                else 0
                            end as other_c,
                            '1' as incoming_cs
                        from
                            soda_international_dwd.dwd_service_ticket_global_d_increment
                        where
                            concat_ws('-', year, month, day) between '2024-01-01'
                            AND '2024-06-30'
                            and country_code in ('MX', 'CO', 'CR', 'PE')
                            and order_id is not null
                            and length(order_id) > 3
                            and is_auto <> '1'
                            and category_id not in (
                                '960043658',
                                '960021416',
                                '960043656',
                                '960022290',
                                '960043482',
                                '960043480',
                                '960043476',
                                '960043570',
                                '960019030',
                                '960043568'
                            )
                            and category_id not in ('960007794', '960007832')
                            and requester_type in ('1')
                    ) as incoming on incoming.order_id = order_base.order_id
                group by
                    order_base.shop_id,
                    order_base.stat_date
            ) as wrong_food_missing_items on shop_basic.shop_id = wrong_food_missing_items.shop_id
            and shop_basic.stat_date = wrong_food_missing_items.stat_date
    ) as base
group by
    country_code,
    MONTH(base.stat_date),
    priority --,priority.potential
order by
    country_code,
    MONTH(base.stat_date) asc,
    priority asc;