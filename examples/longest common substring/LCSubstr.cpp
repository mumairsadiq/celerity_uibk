#include <cstdio>
#include <celerity.h>

int len_str1 = 100, len_str2 = 100;


template <typename T>
void init_zero(celerity::distr_queue& queue, celerity::buffer<T, 1> buf) {
    queue.submit([=](celerity::handler& cgh) {
        auto dw_buf = buf.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
        cgh.parallel_for<class init_zero>(buf.get_range(), [=](cl::sycl::item<1> item) { dw_buf[item[0]] = 0; });
        });
}

void compute_ith_row(celerity::distr_queue queue, celerity::buffer<int, 1> row_pre_buf,
    celerity::buffer<int, 1> row_curr_buf,
    celerity::buffer<int, 1> row_max_buf,
    celerity::buffer<int, 1> row_max_idx_buf,
    celerity::buffer<unsigned char, 1> str2_buf,
    const unsigned char ch, const int iter) {

    queue.submit([=](celerity::handler& cgh) {
        auto a = row_curr_buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<1>());
        auto b = row_pre_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<1>(1));
        auto s2 = str2_buf.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
        auto maxm = row_max_buf.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<1>());
        auto maxm_idx = row_max_idx_buf.get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<1>());
        cgh.parallel_for<class compute_ith_row>(cl::sycl::range<1>(iter), [=](cl::sycl::item<1> item) {


            int j = item[0];
            int val = 0;
            if (j == 0)
            {
                a[j] = val;
            }
            else if (ch == s2[j])
            {
                val = b[j - 1] + 1;
                a[j] = val;

                if (val > maxm[j])
                {
                    maxm[j] = val;
                    maxm_idx[j] = j;
                }
            }
            else
            {
                a[j] = val;
            }

            });
        });
}


int main(int argc, char** argv)
{
    celerity::runtime::init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool is_master = rank == 0;

    if (argc < 4)
    {
        if (is_master)
            printf("Please provide arguements [filename1] [length1] [filename2] [length2]\n");
        return EXIT_FAILURE;
    }


    char* fname1 = argv[1];
    len_str1 = atoi(argv[2]);
    char* fname2 = argv[3];
    len_str2 = atoi(argv[4]);



    unsigned char* str1 = (unsigned char*)malloc((len_str1 + 2) * sizeof(unsigned char));
    unsigned char* str2 = (unsigned char*)malloc((len_str2 + 2) * sizeof(unsigned char));

    FILE* fp1, * fp2;
    fp1 = fopen(fname1, "r");
    if (!fp1)
    {
        printf("Couldn't open file %s\n", fname1);
        return EXIT_FAILURE;
    }

    fp2 = fopen(fname2, "r");
    if (!fp2)
    {
        fclose(fp1);
        printf("Couldn't open file %s\n", fname2);
        return EXIT_FAILURE;
    }


    str1[0] = str2[0] = ' ';
    for (int iu = 1; iu <= len_str1; iu++)
    {
        str1[iu] = (unsigned char)fgetc(fp1);
    }

    for (int iu = 1; iu <= len_str2; iu++)
    {
        str2[iu] = (unsigned char)fgetc(fp2);
    }

    fclose(fp1);
    fclose(fp2);


    str1[len_str1 + 1] = str2[len_str2 + 1] = '\0';
    {
        celerity::distr_queue queue;

        celerity::buffer<unsigned char, 1> str2_buf(str2, cl::sycl::range<1>(len_str2 + 2));

        celerity::buffer<int, 1> row_pre_buf(cl::sycl::range<1>(len_str2 + 1));
        celerity::buffer<int, 1> row_curr_buf(cl::sycl::range<1>(len_str2 + 1));
        celerity::buffer<int, 1> row_max_buf(cl::sycl::range<1>(len_str2 + 1));
        celerity::buffer<int, 1> row_max_idx_buf(cl::sycl::range<1>(len_str2 + 1));
        init_zero<int>(queue, row_pre_buf);
        init_zero<int>(queue, row_max_buf);
        init_zero<int>(queue, row_max_idx_buf);


        for (int i = 1; i <= len_str1; i++)
        {
            if (i % 2 != 0)
                compute_ith_row(queue, row_pre_buf, row_curr_buf, row_max_buf, row_max_idx_buf, str2_buf, str1[i], len_str2 + 1);
            else
                compute_ith_row(queue, row_curr_buf, row_pre_buf, row_max_buf, row_max_idx_buf, str2_buf, str1[i], len_str2 + 1);

        }

        int leng_max = 0;
        auto range = cl::sycl::range<1>(len_str2 + 1);
        queue.submit(celerity::allow_by_ref, [&](celerity::handler& cgh) {
            auto result = row_max_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::one_to_one<1>());
            auto result_idx = row_max_idx_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::one_to_one<1>());
            cgh.host_task(range, [=, &leng_max](celerity::partition<1> part) {

                int idx_max = 0;
                auto sr = part.get_subrange();
                for (size_t i = sr.offset[0]; i < sr.offset[0] + sr.range[0]; ++i)
                {
                    if (result[{i}] > leng_max)
                    {
                        leng_max = result[{i}];
                        idx_max = result_idx[{i}];
                    }
                }

                int all_res[2] = { leng_max, idx_max };
                int local_res[2] = { leng_max, idx_max };
                MPI_Allreduce(local_res, all_res, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
                idx_max = all_res[1];
                leng_max = all_res[0];
                if (is_master)
                {
                    printf("Length of longest common substring is: %d\n", all_res[0]);
                    str2[idx_max + 1] = '\0';
                    printf("Longest common substring is %s\n", &str2[idx_max - leng_max + 1]);
                }
                });
            });


    }
    return EXIT_SUCCESS;
}
