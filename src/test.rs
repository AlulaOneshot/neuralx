#[test]
#[should_panic]
fn neualx_test() {
    let network = crate::Network::new(vec![2, 3, 1]);
    let input = vec![0.5, -0.5];
    let output = network.pass(&input);
    println!("Network output: {:#?}", output);
}