def crop(crop_name):
    crop_data = {
    "black_pepper":["/static/black-pepper.png", "Kerala, Karnataka and Tamil Nadu ", "rabi","USA,Sweden,United Kingdom,Poland,Turkey"],
    "cardamom":["/static/cardamon.jpg", " Kerala, Karnataka,Tamil Nadu ", "kharif","USA, Japan, Middle Eastern Countries such as Saudi Arabia, Kuwait, Qatar, Iraq, Bahrain and Jordan"],
    "chilly":["/static/dry-chilly.jpg", "Andhra Pradesh, Telangana, Karnataka,West Bengal", "kharif & rabi","Vietnam ,Thailand ,Srilanka ,Malaysia ,Bangaldesh"],
    "coriander":["/static/coriander.jpg", "Madhya Pradesh, Gujarat and Rajasthan", "rabi", "Malaysia , UAE , UK , Saudi Arabia "],
    "cumin":["/static/jeeraa.png", "Gujarat and Rajasthan", "rabi", "Vietnam, United Arab Emirates, United States of America, Brazil, Malaysia and Egypt. "],
    "garlic":["/static/garlic.jpg", "Gujarat, Madhya Pradesh, Orissa, Maharashtra, Uttar Pradesh, and Rajasthan","kharif & rabi", "Veitnam, Bangladesh, Iran, Malaysia"],
    "ginger":["/static/dry-ginger.jpeg", "Punjab, Haryana, Maharashtra, Tamil Nadu, Madhya Pradesh, Gujarat","rabi", " China, Bangladesh, Egypt"],
    "rapeseed_mustard":["/static/mustard.jpg", "Uttar Pradesh, Madhya Pradesh, Bihar, West Bengal, Rajasthan,Haryana,Gujarat,Jharkhand,Assam,Punjab", "rabi", "Pakistan, Cyprus,United Arab Emirates"],
    "turmeric":["/static/dried-turmeric.jpeg", "Andhra Pradesh, Tamil Nadu, Orissa, Karnataka, West Bengal, Gujarat and Kerala", "kharif", "the UAE, Iran, the US, Sri Lanka, Japan, the UK, Malaysia , South Africa ,  Myanmar, Indonesia and the Netherlands"]
    
    }
    return crop_data[crop_name]