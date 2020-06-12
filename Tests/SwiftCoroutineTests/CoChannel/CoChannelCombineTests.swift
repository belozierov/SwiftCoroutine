//
//  CoChannelCombineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.06.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import XCTest
import Combine
import SwiftCoroutine
import Foundation

@available(OSX 10.15, iOS 13.0, *)
class CoChannelCombineTests: XCTestCase {
    
    var cancellables = Set<AnyCancellable>()
    
    func testSubscribe() {
        let exp = expectation(description: "testSubscribe")
        exp.expectedFulfillmentCount = 100_001
        let channel = (0..<100_000).publisher.subscribeCoChannel()
        channel.makeIterator().forEach { _ in exp.fulfill() }
        channel.whenComplete { exp.fulfill() }
        wait(for: [exp], timeout: 5)
    }
    
    func testSubscription() {
        let exp = expectation(description: "testSubscription")
        exp.expectedFulfillmentCount = 10_001
        let channel = CoChannel<Int>()
        let publisher = channel.publisher()
        publisher.sink(receiveCompletion: {
            switch $0 {
            case .finished:
                exp.fulfill()
            case .failure(let error):
                XCTFail(error.localizedDescription)
            }
        }, receiveValue: { _ in
            exp.fulfill()
        }).store(in: &cancellables)
        DispatchQueue.global().async {
            for i in 0..<10_000 { channel.offer(i) }
            channel.close()
        }
        wait(for: [exp], timeout: 5)
    }
    
    func testSubscriptionFail() {
        let exp = expectation(description: "testSubscription")
        let channel = CoChannel<Int>()
        let publisher = channel.publisher()
        publisher.sink(receiveCompletion: {
            switch $0 {
            case .finished:
                XCTFail()
            case .failure:
                exp.fulfill()
            }
        }, receiveValue: { _ in
            XCTFail()
        }).store(in: &cancellables)
        channel.cancel()
        wait(for: [exp], timeout: 5)
    }
    
    func testSubscriptionCancel() {
        let channel = CoChannel<Int>()
        let cancellable = channel.publisher()
            .sink(receiveCompletion: { _ in XCTFail() },
                  receiveValue: { _ in XCTFail() })
        cancellable.cancel()
        channel.offer(1)
    }
    
}
#endif
