//
//  CoFutureCombineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 17.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import XCTest
import Combine
import SwiftCoroutine
import Foundation

@available(OSX 10.15, iOS 13.0, *)
class CoFutureCombineTests: XCTestCase {
    
    struct TestError: Error {}
    var cancellables = Set<AnyCancellable>()
    
    func testSubscribe() {
        let exp = expectation(description: "testSubscribe")
        exp.expectedFulfillmentCount = 1000
        for i in 0..<1000 {
            let future = Future<Int, Never> { promise in
                DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(100)) {
                    promise(.success(i))
                }
            }.delay(for: .milliseconds(100), scheduler: DispatchQueue.global()).subscribeCoFuture()
            DispatchQueue.global().startCoroutine {
                XCTAssertEqual(try future.await(), i)
                exp.fulfill()
            }
        }
        wait(for: [exp], timeout: 5)
    }
    
    func testSubscription() {
        let exp = expectation(description: "testSubscription")
        let promise = CoPromise<Int>()
        promise.publisher()
            .map { $0 + 1 }
            .sink(receiveCompletion: {
                switch $0 {
                case .finished: exp.fulfill()
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                }
            }, receiveValue: { XCTAssertEqual($0, 2) })
            .store(in: &cancellables)
        promise.success(1)
        wait(for: [exp], timeout: 1)
    }
    
    func testSubscriptionFail() {
        let exp = expectation(description: "testSubscriptionFail")
        let promise = CoPromise<Int>()
        promise.publisher()
            .sink(receiveCompletion: {
                switch $0 {
                case .failure(_ as TestError):
                    exp.fulfill()
                default: XCTFail()
                }
            }, receiveValue: { _ in XCTFail() })
            .store(in: &cancellables)
        promise.fail(TestError())
        wait(for: [exp], timeout: 1)
    }
    
    func testSubscriptionCancel() {
        let promise = CoPromise<Int>()
        let cancellable = promise.publisher()
            .sink(receiveCompletion: { _ in XCTFail() },
                  receiveValue: { _ in XCTFail() })
        cancellable.cancel()
        promise.success(1)
    }
    
}
#endif
