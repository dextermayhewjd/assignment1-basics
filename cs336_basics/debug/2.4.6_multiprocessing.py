from multiprocessing import Pool, cpu_count



print(cpu_count())


text1 = "low low low low" 
text2 = "low lower lower widest" 
text3 = "widest widest newest newest" 
text4 = "newest newest newest newest"

list_text = [text1,text2,text3,text4]

'''
这里理解 pool.map() 在干嘛
会把pool.map(x,y)
y中每一个元素 
作为实参传给task(x)
'''
# def task1(x:str)->list[str]:
#     # print(x) # 拿到的是 string
#     token_list = x.split(" ")
#     # print(token_list) # 返回的是list
#     return token_list


# if __name__ == "__main__":
#     with Pool(4) as pool:
#         results = pool.map(task1, list_text)
#     # print(results)

# # 此处可见这里返回的是一个list

'''
这里成功最小模拟每个process 分个统计词频
'''
# def task2(x:str)->dict[str,int]:
#     word_list = x.split(" ")
#     dict_word_count = {}
#     for word in word_list:
#         dict_word_count[word] = dict_word_count.get(word,0)+1
#     # print(dict_word_count)
#     return dict_word_count

# if __name__ == "__main__":

#     with Pool(4) as pool:
#         results = pool.map(task2, list_text)
#     print(results)
    
'''
这里看如何更新主dict
修改的电视修改 不用results = 
而是直接iterate
'''
def task2(x)->dict[str,int]:
    word_list = x.split(" ")
    dict_word_count = {}
    for word in word_list:
        dict_word_count[word] = dict_word_count.get(word,0)+1
    # print(dict_word_count)
    return dict_word_count

if __name__ == "__main__":

    main_dict = {}
    with Pool(4) as pool:
        for sub_dict in pool.map(task2, list_text):   
            for key,value in sub_dict.items():
                main_dict[key] = main_dict.get(key,0)+value
    print(main_dict)
    

from pretokenization_example import find_chunk_boundaries

