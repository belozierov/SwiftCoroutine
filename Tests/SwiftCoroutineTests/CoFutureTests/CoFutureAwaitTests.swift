//
//  CoFutureAwaitTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 17.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureAwaitTests: XCTestCase {

    func testOutCoroutineCall() {
        let promise = CoPromise<Int>()
        do {
            _ = try promise.await()
            XCTFail()
        } catch let error as CoroutineError {
            return XCTAssertEqual(error, .mustBeCalledInsideCoroutine)
        } catch {
            XCTFail()
        }
    }

    func testAwait() {
        let exp = expectation(description: "testAwait")
        let promise = CoPromise<Int>()
        DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(1)) {
            promise.success(1)
        }
        CoroutineDispatcher.main.execute {
            XCTAssertEqual(try promise.await(), 1)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 2)
    }
    
    func testConcurrency() {
        var array = Array(repeating: 0, count: 100_000)
        let exp = expectation(description: "testConcurrency")
        exp.expectedFulfillmentCount = array.count
        let queue = DispatchQueue.global()
        DispatchQueue.concurrentPerform(iterations: array.count) { index in
            let promise = CoPromise<Int>()
            queue.asyncAfter(deadline: .now() + .microseconds(index)) {
                promise.success(index)
            }
            queue.asyncAfter(deadline: .now() + .microseconds(array.count - index)) {
                CoroutineDispatcher.global.execute {
                    array[index] = try promise.await()
                    exp.fulfill()
                }
            }
        }
        wait(for: [exp], timeout: 5)
        XCTAssertTrue(array.enumerated().allSatisfy { $0.element == $0.offset })
    }
    
//    func testAbc() {
//        measure {
//            for i in 0..<100_00 {
//                CoroutineTaskExecutor.defaultShared.execute(on: .immediate) {
//                    let result: Int = try! Coroutine.await { completion in
//                        DispatchQueue.global().asyncAfter(deadline: .now() + 0.001) {
//                            completion(i)
//                        }
//                    }
//                    if result != i { fatalError() }
//                }
//            }
//        }
//    }
//
//    func testAbc2() {
//        measure {
//            for i in 0..<100_00 {
//                CoroutineTaskExecutor.defaultShared.execute(on: .immediate) {
//                    let result: Int = try! Coroutine.await2 { completion in
//                        DispatchQueue.global().asyncAfter(deadline: .now() + 0.001) {
//                            completion(i)
//                        }
//                    }
//                    if result != i { fatalError() }
//                }
//            }
//        }
//    }
    
    func testFoo() {
        measure {
            for i in 0..<10_000 {
                let promise = CoPromise<Int>()
                CoroutineTaskExecutor.defaultShared.execute(on: .immediate) {
                    XCTAssertEqual(try? promise.await(), i)
                }
                promise.success(i)
            }
        }
    }
    
    func testFoo2() {
        measure {
            for i in 0..<10_000 {
                let promise = CoPromise<Int>()
                CoroutineTaskExecutor.defaultShared.execute(on: .immediate) {
                    XCTAssertEqual(try? promise.await(), i)
                }
                promise.success(i)
            }
        }
    }

}
