import torch
# name
user_sparse_feat = ['phone_price','phone_model','phone_brand','aby','pred_career_type','life_stage','activity','pred_car_brand','car_brand_name','pred_car_series','car_line_name','house','phone_price_level_prefer','phone_use_month','hild','pred_life_stage_married','pred_is_undergraduate','car','pred_education_degree','pred_gender','is_cap','city_level','county_name','is_foreign','nation_id','city_name','prov_name','vip_level_name','id_age','id_age_level','id_birthyear','id_gender','buyer_star_name',"pp_level"]
user_dense_feat = ['clt_slr_cnt_1y','cart_itm_cnt_2w','ipv_1m','pay_ord_cnt_3m','clt_itm_cnt_2w','pay_ord_cnt_6m','pay_ord_amt_6m','pay_ord_days_1m','clt_slr_cnt_2w','clt_itm_cnt_1w','clt_itm_cnt_1m','pv_1m','clt_slr_cnt_1m','stay_time_len_1m','pay_ord_amt_1y','pay_ord_cnt_1y','vst_days_1m','cart_itm_cnt_6m','vst_slr_cnt_1m','clt_slr_cnt_6m','pay_ord_itm_cnt_6m','cart_itm_cnt_3m','pay_ord_itm_qty_1m','clt_itm_cnt_1y','pay_ord_itm_qty_1y','clt_itm_cnt_6m','pay_ord_itm_qty_3m','clt_slr_cnt_1w','pay_ord_itm_qty_6m','pay_ord_amt_1m','pay_ord_amt_3m','pay_ord_cnt_1m','pay_ord_days_1y','pay_ord_days_3m','cart_itm_cnt_1m','pay_ord_days_6m','vst_cate_lvl1_cnt_1m','pay_ord_itm_cnt_1m','vst_itm_cnt_1m','pay_ord_itm_cnt_1y','cart_itm_cnt_1w','pay_ord_itm_cnt_3m','pp_level_pay_rank_score','pp_level_click_rank_score','pp_level_profile_rank_score','pp_level_final_rank_score','pp_level_pay_quantity','pp_level_pay_cnt','pp_level_pay_fee','pp_level_pay_hdj','pp_level_pay_leafcat_cnt','pp_level_click_leafcat_cnt','pp_level_click_item_price','pp_level_click_cnt','pp_level_fee_sum','pp_level_fee_all','pp_level_pay_fee_rank_score']
seq_feat = ["shoutao_stay_time_seq","shoutao_ipv_seq","shoutao_ipv_stay_time_seq","shoutao_pay_cnt_seq","shoutao_pay_qty_seq","shoutao_pay_amt_seq","shoutao_cart_cnt_seq","shoutao_cart_qty_seq","shoutao_collect_cnt_seq","shoutao_pure_ord_cnt_seq","shoutao_pure_ord_qty_seq","shoutao_pure_ord_amt_seq","shoutao_is_dau_seq","shoutao_is_active_dau_seq","shoutao_start_count_1d_seq","ltv_cfm_ord_cnt","ltv_cfm_ord_itm_qty","ltv_cfm_ord_amt","ltv_yj_amt","ltv_b_cfm_ord_cnt","ltv_b_cfm_ord_itm_qty","ltv_b_cfm_ord_amt","ltv_c_cfm_ord_cnt","ltv_c_cfm_ord_itm_qty","ltv_c_cfm_ord_amt","ltv_tm_cate_yj_amt","ltv_soft_fee_amt","ltv_total_ad_cost","ltv_se_ad_cost","ltv_sc_ad_cost","ltv_gh_ad_cost","shoucai_is_active","shoucai_pv","shoucai_dpv1_clk","shoucai_ipv","shoucai_nd_ipv","shoucai_detail_ipv","shoucai_pay_amt","shoucai_pay_cnt","shoucai_pay_amt1","shoucai_pay_cnt1","shoucai_pure_pay_amt","shoucai_pure_pay_cnt","shoucai_pure_pay_amt1","shoucai_pure_pay_cnt1","shoucai_total_stay_time","shoucai_ad_pv","shoucai_ad_ipv","gouhou_pv","gouhou_dpv1_clk","gouhou_ipv","gouhou_nd_ipv","gouhou_detail_ipv","gouhou_pay_amt","gouhou_pay_cnt","gouhou_pay_amt1","gouhou_pay_cnt1","gouhou_pure_pay_amt","gouhou_pure_pay_cnt","gouhou_pure_pay_amt1","gouhou_pure_pay_cnt1","gouhou_total_stay_time","gouhou_ad_pv","gouhou_ad_ipv","search_imps_1d","search_direct_lead_ipv_1d","search_direct_pure_lead_ipv_1d","search_direct_lead_pay_ord_amt_1d","search_direct_lead_pay_ord_cnt_1d","search_direct_lead_pay_ord_itm_qty_1d","search_direct_pure_lead_pay_ord_amt_1d","search_direct_pure_lead_pay_ord_cnt_1d","search_direct_pure_lead_pay_ord_itm_qty_1d","search_direct_lead_cart_amt_1d","search_direct_lead_cart_cnt_1d","search_direct_lead_cart_itm_qty_1d","search_lead_ipv_1d","search_pure_lead_ipv_1d","search_lead_pay_ord_amt_1d","search_lead_pay_ord_cnt_1d","search_lead_pay_ord_itm_qty_1d","search_pure_lead_pay_ord_amt_1d","search_pure_lead_pay_ord_cnt_1d","search_pure_lead_pay_ord_itm_qty_1d","search_lead_cart_amt_1d","search_lead_cart_cnt_1d","search_lead_cart_itm_qty_1d","search_p4p_imps_1d","search_p4p_direct_lead_ipv_1d","search_p4p_direct_pure_lead_ipv_1d","search_p4p_direct_lead_pay_ord_amt_1d","search_p4p_direct_lead_pay_ord_cnt_1d","search_p4p_direct_lead_pay_ord_itm_qty_1d","search_p4p_direct_pure_lead_pay_ord_amt_1d","search_p4p_direct_pure_lead_pay_ord_cnt_1d","search_p4p_direct_pure_lead_pay_ord_itm_qty_1d","search_p4p_direct_lead_cart_amt_1d","search_p4p_direct_lead_cart_cnt_1d","search_p4p_direct_lead_cart_itm_qty_1d","search_p4p_lead_ipv_1d","search_p4p_pure_lead_ipv_1d","search_p4p_lead_pay_ord_amt_1d","search_p4p_lead_pay_ord_cnt_1d","search_p4p_lead_pay_ord_itm_qty_1d","search_p4p_pure_lead_pay_ord_amt_1d","search_p4p_pure_lead_pay_ord_cnt_1d","search_p4p_pure_lead_pay_ord_itm_qty_1d","search_p4p_lead_cart_amt_1d","search_p4p_lead_cart_cnt_1d","search_p4p_lead_cart_itm_qty_1d","search_nature_imps_1d","search_nature_direct_lead_ipv_1d","search_nature_direct_pure_lead_ipv_1d","search_nature_direct_lead_pay_ord_amt_1d","search_nature_direct_lead_pay_ord_cnt_1d","search_nature_direct_lead_pay_ord_itm_qty_1d","search_nature_direct_pure_lead_pay_ord_amt_1d","search_nature_direct_pure_lead_pay_ord_cnt_1d","search_nature_direct_pure_lead_pay_ord_itm_qty_1d","search_nature_direct_lead_cart_amt_1d","search_nature_direct_lead_cart_cnt_1d","search_nature_direct_lead_cart_itm_qty_1d","search_nature_lead_ipv_1d","search_nature_pure_lead_ipv_1d","search_nature_lead_pay_ord_amt_1d","search_nature_lead_pay_ord_cnt_1d","search_nature_lead_pay_ord_itm_qty_1d","search_nature_pure_lead_pay_ord_amt_1d","search_nature_pure_lead_pay_ord_cnt_1d","search_nature_pure_lead_pay_ord_itm_qty_1d","search_nature_lead_cart_amt_1d","search_nature_lead_cart_cnt_1d","search_nature_lead_cart_itm_qty_1d","search_qzt_imps_1d","search_qzt_direct_lead_ipv_1d","search_qzt_direct_pure_lead_ipv_1d","search_qzt_direct_lead_pay_ord_amt_1d","search_qzt_direct_lead_pay_ord_cnt_1d","search_qzt_direct_lead_pay_ord_itm_qty_1d","search_qzt_direct_pure_lead_pay_ord_amt_1d","search_qzt_direct_pure_lead_pay_ord_cnt_1d","search_qzt_direct_pure_lead_pay_ord_itm_qty_1d","search_qzt_lead_ipv_1d","search_qzt_pure_lead_ipv_1d","search_qzt_lead_pay_ord_amt_1d","search_qzt_lead_pay_ord_cnt_1d","search_qzt_lead_pay_ord_itm_qty_1d","search_qzt_pure_lead_pay_ord_amt_1d","search_qzt_pure_lead_pay_ord_cnt_1d","search_qzt_pure_lead_pay_ord_itm_qty_1d","search_lead_stay_time_1d","search_is_active_dau"]
label = ['label_ltv', 'label_cost', 'label_pay_amt', 'label_yj_amt', 'label_active_dau']
treatment = ['ab_tag']
selected_short = ['shoutao_stay_time_seq_short', 'shoutao_ipv_seq_short', 'shoutao_pay_cnt_seq_short', 'shoutao_is_active_dau_seq_short', 'shoucai_ipv_short', 'shoucai_is_active_short', 'search_is_active_dau_short', 'search_lead_stay_time_1d_short', 'ltv_total_ad_cost_short', 'ltv_yj_amt_short']

# path
exp_file_path = 'data/ltv_exp_data_100_sample.csv'
obs_file_path = 'data/ltv_obs_data_100_sample.csv' 

# train 
device = torch.device("cuda:7") 
lamb = 1e-4
batch_size = 256
learning_rate = 0.0005
num_epochs = 50
hidden_dim = 256
base = 0
exp = 1
exp_num = 5

# dim
S_window = 7
history_seq_feature_dim = 180
s_dim = 10 * S_window
x_dim = 57
x_sparse_dim = 34
x_seq_dim = 152
y_dim = 1
sparse_embedding_dim = 16
s_seq_dim = 256
embed_dim = x_seq_dim
fc_layers = [embed_dim * 180, 512, s_seq_dim]  
num_heads = 8